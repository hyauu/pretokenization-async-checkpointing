from timing import raw_ns, raw_seconds
import time
import torch
from torchtune.training import scale_grads_

def update(model,opt,loss_fn,iterator,expected,accumulation,saver=None,events=False,on_first_forward=None):
    started=raw_seconds(); fetch_s=0.;total_tokens=0;running_loss=0.;num_tokens=0
    first=torch.cuda.Event(enable_timing=True) if events else None
    last=torch.cuda.Event(enable_timing=True) if events else None
    if first:first.record()
    for index in range(accumulation):
        tf=raw_seconds();batch=next(iterator);fetch_s+=raw_seconds()-tf
        entry=expected[index]
        assert batch['sample_ids']==entry['sample_ids']
        assert batch['effective_tokens']==entry['effective_tokens']
        assert all(e==entry['epoch'] for e in batch['epochs'])
        total_tokens+=batch['effective_tokens']
        tokens=batch['tokens'].cuda();labels=batch['labels'].cuda()
        current_tokens=(labels!=-100).sum();num_tokens+=current_tokens
        if index==0 and on_first_forward is not None:on_first_forward()
        hidden=model(tokens);current_loss=loss_fn(hidden,labels)*current_tokens
        running_loss+=current_loss.detach();current_loss.backward()
        del hidden,current_loss,tokens,labels,batch
    # Training never mutates persistent model buffers (checked before any run).
    # Parameters and Adam state stay unchanged until this staging fence.
    tw=raw_seconds()
    if saver:saver.wait_staging()
    stage_wait_s=raw_seconds()-tw
    scale_grads_(model.parameters(),1.0/num_tokens)
    opt.step();opt.zero_grad(set_to_none=True)
    if last:last.record()
    # Preserve the original recipe's per-update loss.item() synchronization.
    loss_value=(running_loss/num_tokens).item()
    return {'start':started,'end':raw_seconds(),'fetch_s':fetch_s,'staging_wait_s':stage_wait_s,
            'tokens':total_tokens,'loss':loss_value,'cuda_events':(first,last) if events else None}

def persistent_buffer_keys(model):
    params=set(dict(model.named_parameters()).keys())
    return [key for key in model.state_dict() if key not in params]

def require_exhausted(iterator):
    try:next(iterator)
    except StopIteration:return
    raise RuntimeError('Finite data window overflow')
