from timing import raw_ns, raw_seconds
"""Narrow DCP adapter: isolated directories, complete-state capture, explicit commit."""
import concurrent.futures, copy, json, os, pickle, time
from pathlib import Path
import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.staging import DefaultStager, StagingOptions
from torch.distributed.checkpoint.default_planner import DefaultSavePlanner
from torch.distributed.checkpoint.state_dict_saver import AsyncCheckpointerType
from common import ROOT, rng_state, restore_rng, init_adam_state

def atomic_complete(path,value):
    tmp=path/'COMPLETE.tmp'
    with tmp.open('x') as f:json.dump(value,f,sort_keys=True);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path/'COMPLETE')
    fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)
class Saver:
    def __init__(self,run_id,threads,protocol_id,extra_metadata=None):
        self.base=ROOT/'checkpoints'/run_id;self.base.mkdir(parents=True,exist_ok=False)
        self.threads=threads;self.protocol_id=protocol_id;self.stager=DefaultStager(StagingOptions(True,True,True,True))
        self.planner=DefaultSavePlanner(enable_plan_caching=True);self.upload=None;self.staging=None;self.commit=None
        self.events=[];self.commits=0;self.extra_metadata=extra_metadata
    def wait_staging(self):
        if self.staging is not None:
            t=raw_seconds();self.staging.result();self.events.append({'event':'staging_wait','start':t,'end':raw_seconds()});self.staging=None
    def drain(self):
        t=raw_seconds()
        self.wait_staging()
        if self.upload is not None:
            self.upload.result();self.commit.result();self.upload=None;self.commit=None
        self.events.append({'event':'upload_wait','start':t,'end':raw_seconds()})
    def save(self,model,optimizer,step,cursor,manifest_sha,initial=False):
        self.drain()
        path=self.base/f'step_{step:08d}'
        if path.exists():raise RuntimeError('Refusing checkpoint directory reuse')
        meta={'format_version':1,'global_step':step,'committed_batch_cursor':cursor,'manifest_sha256':manifest_sha,
              'protocol_id':self.protocol_id,'rng':rng_state(),'accumulation_boundary':0}
        if self.extra_metadata is not None:meta['extra']=self.extra_metadata()
        # All CPU metadata is serialized before training can mutate any state.
        state={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'metadata':pickle.dumps(meta,protocol=5)}
        started=raw_seconds();commit=concurrent.futures.Future()
        response=dcp.async_save(state,storage_writer=dcp.FileSystemWriter(path,single_file_per_rank=True,sync_files=True,thread_count=self.threads,overwrite=False),
                                planner=self.planner,async_checkpointer_type=AsyncCheckpointerType.PROCESS,async_stager=self.stager)
        self.staging=response.staging_completion;self.upload=response.upload_completion;self.commit=commit
        self.events.append({'event':'save_call','step':step,'initial':initial,'start':started,'end':raw_seconds(),'path':str(path)})
        response.staging_completion.add_done_callback(lambda f:self.events.append({'event':'staging_done','step':step,'time':raw_seconds(),'success':f.exception() is None}))
        def finish(future):
            try:
                future.result()
                atomic_complete(path,{'format_version':1,'global_step':step,'committed_batch_cursor':cursor,'manifest_sha256':manifest_sha,
                                      'protocol_id':self.protocol_id,'initial':initial})
                self.events.append({'event':'commit','step':step,'initial':initial,'start':started,'end':raw_seconds(),'path':str(path)})
                self.commits+=1;commit.set_result(str(path))
            except BaseException as exc:commit.set_exception(exc)
        response.upload_completion.add_done_callback(finish)
        return path
    def close(self):self.drain();self.stager.close()

def complete_candidates(base):
    result=[]
    for p in Path(base).glob('step_*'):
        if (p/'COMPLETE').is_file() and (p/'.metadata').is_file():
            flag=json.loads((p/'COMPLETE').read_text());result.append((flag['global_step'],p))
    return [p for _,p in sorted(result,reverse=True)]
def load_path(path,model,optimizer,expected_manifest):
    flag=json.loads((Path(path)/'COMPLETE').read_text())
    if flag['manifest_sha256']!=expected_manifest:raise RuntimeError('Manifest mismatch')
    init_adam_state(optimizer)
    state={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'metadata':b''}
    dcp.load(state,checkpoint_id=path)
    model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
    meta=pickle.loads(state['metadata'])
    if meta['global_step']!=flag['global_step'] or meta['committed_batch_cursor']!=flag['committed_batch_cursor']:raise RuntimeError('State/commit mismatch')
    restore_rng(meta['rng']);return meta
