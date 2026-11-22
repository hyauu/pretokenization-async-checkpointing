"""Guest logical CPU placement and readback. Standard library/psutil only."""
import json,os,time
from pathlib import Path
import psutil
from timing import raw_ns

def start_ticks(pid):
    text=Path(f'/proc/{pid}/stat').read_text()
    return int(text[text.rfind(')')+2:].split()[19])

def identity(pid):
    p=psutil.Process(pid)
    return dict(pid=pid,ppid=p.ppid(),create_time=p.create_time(),start_ticks=start_ticks(pid),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),pgid=os.getpgid(pid),cmdline=p.cmdline())

def affinity_snapshot(pid,expected):
    rows=[];unreadable=[];exited=[]
    for d in sorted(Path(f'/proc/{pid}/task').glob('[0-9]*'),key=lambda x:int(x.name)):
        tid=int(d.name)
        try:rows.append(dict(tid=tid,actual=sorted(os.sched_getaffinity(tid)),expected=list(expected)))
        except ProcessLookupError:exited.append(tid)
        except (PermissionError,OSError) as e:unreadable.append(dict(tid=tid,error=repr(e)))
    return dict(pid=pid,raw_ns=raw_ns(),threads=rows,exited_tids=exited,unreadable=unreadable,
                violations=[r for r in rows if r['actual']!=list(expected)],readable=bool(rows))

def set_all_tids(pid,mask):
    """Called only outside measurement; repeat enumeration covers creation races."""
    seen=set()
    for _ in range(8):
        tids={int(d.name) for d in Path(f'/proc/{pid}/task').glob('[0-9]*')}
        for tid in tids:
            try:os.sched_setaffinity(tid,mask)
            except ProcessLookupError:pass
        check=affinity_snapshot(pid,mask)
        if tids==seen and check['readable'] and not check['violations'] and not check['unreadable']:return check
        seen=tids
    raise RuntimeError('Placement setup could not verify all live TIDs: '+str(check))

def persist(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');os.replace(tmp,path)

def prepare(request,out,saver_present):
    train=identity(os.getpid());writer=None;changed=None
    assert train['pgid']==train['pid'],'Training must own isolated task PGID'
    before=affinity_snapshot(train['pid'],request['training_cpus'])
    assert before['readable'] and not before['violations'] and not before['unreadable'],before
    if saver_present:
        from torch.distributed.checkpoint import _async_process_executor as executor
        owner=executor._CHECKPOINT_PROCESS
        assert owner is not None and owner._save_process.is_alive()
        writer=identity(owner._save_process.pid)
        assert writer['ppid']==train['pid'] and writer['pgid']==train['pgid']
        assert writer['start_ticks']>=train['start_ticks'] and writer['boot_id']==train['boot_id'] and writer['pid']!=train['pid']
        assert 'resource_tracker' not in ' '.join(writer['cmdline'])
        changed=set_all_tids(writer['pid'],request['writer_cpus'])
    else:
        assert request['placement']=='absent' and request['writer_cpus']==[]
    value=dict(attempt_id=request['attempt_id'],placement=request['placement'],training=train,writer=writer,
               training_cpus=request['training_cpus'],writer_cpus=request['writer_cpus'],setup_raw_ns=raw_ns(),
               training_setup=before,writer_setup=changed,
               identity_source='_CHECKPOINT_PROCESS._save_process.pid after initial save+drain' if writer else 'no Saver constructed',
               scope='Guest logical CPU masks; no claim of physical core/SMT/cache/IO isolation')
    persist(out/'placement_ready.json',value)
    return value

def endpoint(value,label,out):
    checks=dict(label=label,raw_ns=raw_ns(),training=affinity_snapshot(value['training']['pid'],value['training_cpus']),
                writer=affinity_snapshot(value['writer']['pid'],value['writer_cpus']) if value['writer'] else None)
    persist(out/f'placement_{label}.json',checks)
    for row in (checks['training'],checks['writer']):
        if row:assert row['readable'] and not row['violations'] and not row['unreadable'],row
    return checks

def process_sample(who,mask,include_affinity=False):
    if who is None:return dict(status='ABSENT')
    pid=who['pid'];start=raw_ns()
    try:
        p=psutil.Process(pid)
        actual_ticks=start_ticks(pid)
        assert actual_ticks==who['start_ticks'],('PID reused',who['start_ticks'],actual_ticks)
        cpu=p.cpu_times();row=dict(status='OBSERVED',pid=pid,start_ticks=actual_ticks,create_time=p.create_time(),raw_ns=start,user_s=cpu.user,system_s=cpu.system)
        try:row['io']=p.io_counters()._asdict()
        except (AttributeError,NotImplementedError,psutil.AccessDenied,psutil.NoSuchProcess) as e:row['io']=None;row['io_gap']=repr(e)
        threads=[]
        for d in Path(f'/proc/{pid}/task').glob('[0-9]*'):
            try:
                vals=[int(x) for x in (d/'schedstat').read_text().split()]
                status=(d/'status').read_text().splitlines()
                cs={s.split(':')[0]:int(s.split(':')[1]) for s in status if s.startswith(('voluntary_ctxt_switches:','nonvoluntary_ctxt_switches:'))}
                threads.append(dict(tid=int(d.name),run_ns=vals[0],runqueue_wait_ns=vals[1],timeslices=vals[2],**cs))
            except (FileNotFoundError,ProcessLookupError,PermissionError,OSError):pass
        row['threads']=threads
        row['schedstat_interpretation']='Runnable runqueue wait only; absent/all-zero disabled values are NA, never evidence of no contention'
        if include_affinity:row['affinity']=affinity_snapshot(pid,mask)
        return row
    except (psutil.NoSuchProcess,ProcessLookupError):return dict(status='EXITED',pid=pid,raw_ns=start)
    except (psutil.AccessDenied,OSError) as e:return dict(status='UNREADABLE',pid=pid,raw_ns=start,error=repr(e))
