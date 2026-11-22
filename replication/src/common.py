from timing import raw_ns, raw_seconds
import copy, hashlib, json, os, pickle, random, socket, time
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, Sampler
from torchtune.data import AlpacaToMessages, OpenAIToMessages, padded_collate_sft
from torchtune.datasets._sft import SFTTransform
from torchtune.models.qwen3 import qwen3_0_6b_instruct, qwen3_tokenizer
from torchtune.models.qwen3._convert_weights import qwen3_hf_to_tune
from torchtune.modules.loss import LinearCrossEntropyLoss
from safetensors.torch import load_file

ROOT=Path(os.environ.get('S09_ROOT','/opt/s09_cpu_allocation_20261023'))
MODEL_ROOT=Path('/opt/s09/data/model')  # verified prior-round asset, read only
WORKLOAD=os.environ.get('S09_WORKLOAD','U')
if WORKLOAD not in ('A','U'):raise ValueError('Unknown workload')
DATA_ROOT=Path(os.environ.get('S09_DATA_ROOT','/opt/s09_followup_20261021/raw_v2/data/ultrachat'))
def dump(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
def filehash(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        while chunk:=f.read(8*1024*1024):h.update(chunk)
    return h.hexdigest()
def seed_all(seed):random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
def tokenizer(max_seq_len=1024):
    return qwen3_tokenizer(path=str(MODEL_ROOT/'vocab.json'),merges_file=str(MODEL_ROOT/'merges.txt'),max_seq_len=max_seq_len)
def transform(max_seq_len=1024):
    messages=AlpacaToMessages(train_on_input=True) if WORKLOAD=='A' else OpenAIToMessages(masking_strategy='train_on_all')
    return SFTTransform(message_transform=messages,model_transform=tokenizer(max_seq_len))
def init_pg():
    with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
    os.environ.update(MASTER_ADDR='127.0.0.1',MASTER_PORT=str(port),RANK='0',LOCAL_RANK='0',WORLD_SIZE='1')
    dist.init_process_group('gloo',init_method='env://',rank=0,world_size=1)
def build_model(seed):
    seed_all(seed)
    old=torch.get_default_dtype();torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device('cuda'):model=qwen3_0_6b_instruct()
    finally:torch.set_default_dtype(old)
    state=qwen3_hf_to_tune(load_file(str(MODEL_ROOT/'model.safetensors'),device='cpu'),num_heads=16,num_kv_heads=8,dim=1024,head_dim=128,tie_word_embeddings=True)
    model.load_state_dict(state,strict=True);del state
    model.train();optimizer=torch.optim.AdamW(model.parameters(),lr=2e-5,fused=True)
    loss=LinearCrossEntropyLoss();loss.set_model_output(model)
    return model,optimizer,loss
def init_adam_state(optimizer):
    for group in optimizer.param_groups:
        for p in group['params']:
            if not optimizer.state[p]:
                optimizer.state[p]={'step':torch.zeros((),dtype=torch.float32,device=p.device),
                                    'exp_avg':torch.zeros_like(p),'exp_avg_sq':torch.zeros_like(p)}
def rng_state():
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch_cpu':torch.get_rng_state().clone(),
            'torch_cuda':[x.clone() for x in torch.cuda.get_rng_state_all()]}
def restore_rng(state):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch_cpu']);torch.cuda.set_rng_state_all(state['torch_cuda'])
def tree_digest(tree):
    h=hashlib.sha256()
    def rec(x):
        if isinstance(x,torch.Tensor):
            t=x.detach().cpu().contiguous();h.update(str(t.dtype).encode());h.update(str(tuple(t.shape)).encode());h.update(memoryview(t.reshape(-1).view(torch.uint8).numpy()))
        elif isinstance(x,dict):
            for k in sorted(x,key=str):h.update(str(k).encode());rec(x[k])
        elif isinstance(x,(list,tuple)):
            for y in x:rec(y)
        elif isinstance(x,np.ndarray):h.update(x.tobytes())
        else:h.update(repr(x).encode())
    rec(tree);return h.hexdigest()
def batch_digest(batch):return tree_digest({k:v for k,v in batch.items() if k in ['tokens','labels','mask','sample_ids','epochs']})

class InputDataset(Dataset):
    """Both pathways keep the same raw and preprocessed assets resident."""
    def __init__(self,mode):self.mode=mode;self._load_assets();self.converter=None
    def _load_assets(self):
        self.records=json.loads((DATA_ROOT/('alpaca.json' if WORKLOAD=='A' else 'raw.json')).read_text())
        self.tokens=np.load(DATA_ROOT/'cache_tokens.npy');self.labels=np.load(DATA_ROOT/'cache_labels.npy')
        self.masks=np.load(DATA_ROOT/'cache_masks.npy');self.offsets=np.load(DATA_ROOT/'cache_offsets.npy')
        self.source_ids=list(range(len(self.records))) if WORKLOAD=='A' else np.load(DATA_ROOT/'source_row_ids.npy').tolist()
        self.local_index={value:index for index,value in enumerate(self.source_ids)}
    def __len__(self):return len(self.records)
    def __getstate__(self):return {'mode':self.mode}
    def __setstate__(self,state):self.mode=state['mode'];self._load_assets();self.converter=None
    def __getitem__(self,index):
        sample_id,epoch=index
        audit=os.environ.get('S09_DATA_AUDIT_DIR')
        if audit:
            with open(Path(audit)/f'data_{os.getpid()}.jsonl','a') as f:
                f.write(json.dumps({'sample_id':sample_id,'epoch':epoch,'start':raw_seconds(),'pid':os.getpid()})+'\n')
        local_id=self.local_index[sample_id]
        if self.mode=='online':
            if self.converter is None:self.converter=transform()
            item=self.converter(copy.deepcopy(self.records[local_id]))
        else:
            a,b=self.offsets[local_id:local_id+2]
            item={'tokens':self.tokens[a:b].tolist(),'labels':self.labels[a:b].tolist(),'mask':self.masks[a:b].tolist()}
        item['sample_id']=sample_id;item['epoch']=epoch
        return item
class WindowBatchSampler(Sampler):
    def __init__(self,entries):self.entries=entries
    def __iter__(self):
        # Freeze the finite list at iterator creation; no lookahead beyond this window.
        for entry in tuple(self.entries):yield [(i,entry['epoch']) for i in entry['sample_ids']]
    def __len__(self):return len(self.entries)
def collate(batch):
    result=padded_collate_sft(batch,padding_idx=151643,ignore_idx=-100)
    result['sample_ids']=[x['sample_id'] for x in batch];result['epochs']=[x['epoch'] for x in batch]
    result['effective_tokens']=int((result['labels']!=-100).sum())
    return result
def worker_init(worker_id):
    worker=torch.utils.data.get_worker_info();worker.dataset.converter=transform()
def make_loader(mode,workers,entries,seed):
    ds=InputDataset(mode);sampler=WindowBatchSampler(entries);gen=torch.Generator().manual_seed(seed+999_999)
    opts={}
    if workers:opts=dict(prefetch_factor=2,persistent_workers=True,multiprocessing_context='spawn',worker_init_fn=worker_init)
    loader=DataLoader(ds,batch_sampler=sampler,collate_fn=collate,num_workers=workers,generator=gen,**opts)
    return loader,sampler
def stop_loader(loader):
    it=getattr(loader,'_iterator',None)
    if it is not None:it._shutdown_workers()
def manifest(seed,num_updates,microbatch=2,warmup_updates=3):
    if WORKLOAD!='A':raise RuntimeError('UltraChat manifests must use full-split row IDs and the frozen prepared subset')
    offsets=np.load(DATA_ROOT/'cache_offsets.npy');labels=np.load(DATA_ROOT/'cache_labels.npy');n=len(offsets)-1
    counts=np.array([np.count_nonzero(labels[offsets[i]:offsets[i+1]]!=-100) for i in range(n)])
    accumulation=16//microbatch;need=(warmup_updates+num_updates)*16;entries=[];epoch=0;consumed=0
    while consumed<need:
        perm=np.random.default_rng(seed+epoch).permutation(n)
        for start in range(0,n-microbatch+1,microbatch):
            ids=perm[start:start+microbatch].tolist();entries.append({'epoch':epoch,'sample_ids':ids,'effective_tokens':int(counts[ids].sum())});consumed+=microbatch
            if consumed>=need:break
        epoch+=1
    warm=warmup_updates*accumulation
    value={'seed':seed,'microbatch':microbatch,'accumulation':accumulation,'warmup_updates':warmup_updates,'measurement_updates':num_updates,
           'warmup':entries[:warm],'measurement':entries[warm:],'effective_supervised_tokens':sum(x['effective_tokens'] for x in entries[warm:]),
           'shuffle_rule':'numpy.default_rng(seed+epoch).permutation(all rows); natural epochs; finite windows'}
    value['manifest_sha256']=hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest();return value
