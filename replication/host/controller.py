import datetime,hashlib,json,msvcrt,os,pathlib,statistics,subprocess,sys,threading,time,traceback
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'host_deps'))
import psutil
from runtime import R,G,GH,DISTRO,F,qpc,utc,save,append,guest,Monitoring
def read(p):return json.loads(pathlib.Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def pause():return (R/'control/USER_PAUSED').exists()
def main():
 lock=(R/'host/controller.lock').open('a+b');lock.seek(0);lock.write(b'1');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
 window=read(R/'execution_window.json');stop_at=datetime.datetime.fromisoformat(window['training_stop_utc'].replace('Z','+00:00')).timestamp()
 stop_qpc=qpc()+max(0,stop_at-time.time())*F
 protocol=read(R/'protocol.json');p_sha=sha(R/'protocol.json');ledger=R/'evidence/gpu_ledger.jsonl'
 session='session_001';monitor=Monitoring();heartbeat_stop=threading.Event()
 receipts=dict(pid=os.getpid(),created_time=psutil.Process().create_time(),command=psutil.Process().cmdline(),started_utc=utc(),protocol_sha256=p_sha,qpc=qpc(),qpc_frequency=F)
 save(R/'host/controller_receipt.json',receipts)
 state=read(R/'state.json');state.update(status='CLOCK_CHECK',controller=receipts,training_started=False)
 def status(s,**kw):state.update(status=s,updated_utc=utc(),**kw);save(R/'state.json',state)
 def heartbeat():
  while not heartbeat_stop.is_set():save(R/'host/heartbeat.json',dict(pid=os.getpid(),created_time=receipts['created_time'],utc=utc(),qpc=qpc()));heartbeat_stop.wait(5)
 hb=threading.Thread(target=heartbeat,daemon=True);hb.start()
 import ctypes
 ctypes.windll.kernel32.SetThreadExecutionState(0x80000003)
 try:
  monitor.start();status('CLOCK_CHECK')
  for _ in range(65):
   if pause():status('USER_PAUSED');return
   if monitor.errors:raise RuntimeError(monitor.errors)
   time.sleep(1)
  qualification=monitor.qualify();boot=qualification['boot_id']
  save(R/'session.json',dict(session_id=session,boot_id=boot,measurement_segment=1,protocol_sha256=p_sha,clock_qualification=qualification,utc=utc()))
  records=[json.loads(x) for x in ledger.read_text().splitlines()] if ledger.exists() else []
  valid={x['slot_id']:x for x in records if x.get('status')=='VALID'}
  assert not valid,'Resuming a previously started controller requires explicit reconstruction of session and attempts'
  failed_signatures=[]
  for slot in protocol['fixed_queue']:
   if pause():status('USER_PAUSED',active_attempt=None);return
   if time.time()+700>=stop_at or qpc()+700*F>=stop_qpc:status('BUDGET_COLLECTION_CLOSED',active_attempt=None);break
   assert sha(R/'protocol.json')==p_sha,'Frozen protocol changed'
   for attempt_number in (1,2,3):
    if pause():status('USER_PAUSED',active_attempt=None);return
    status('PRELAUNCH',next_slot=slot['slot_id'],active_attempt=None,valid_runs=len(valid))
    waiting_start=time.time();samples=[]
    while True:
     if pause():status('USER_PAUSED');return
     if time.time()+700>=stop_at or qpc()+700*F>=stop_qpc:status('BUDGET_COLLECTION_CLOSED');return
     if monitor.errors:raise RuntimeError(monitor.errors)
     assert psutil.disk_usage(str(R)).free>45*1024**3,'Insufficient C free space for bounded current attempt'
     row=monitor.gpu[-1] if monitor.gpu else {};assert row and (qpc()-row['qpc'])/F<5,'GPU monitor stale';temp=float(row.get('temperature.gpu','nan'));util=float(row.get('utilization.gpu','nan'))
     samples.append(dict(utc=utc(),qpc=qpc(),temperature=temp,utilization=util));samples=samples[-3:]
     if len(samples)==3 and all(x['temperature']<=50 and x['utilization']<20 for x in samples) and max(x['temperature'] for x in samples)-min(x['temperature'] for x in samples)<=2:break
     if time.time()-waiting_start>600:
      save(R/'evidence/resource_wait.json',dict(status='RESOURCE_WAIT',since_utc=datetime.datetime.fromtimestamp(waiting_start,datetime.timezone.utc).isoformat(),samples=samples));status('RESOURCE_WAIT')
     time.sleep(5)
    attempt=slot['slot_id']+f'_a{attempt_number}'
    req={**slot,'round_id':protocol['round_id'],'action':'run','attempt_id':attempt,'attempt_number':attempt_number,'protocol':G+'/protocol.json','protocol_sha256':p_sha,'execution_protocol_sha256':sha(R/'execution_protocol.json'),'session_id':session,'measurement_segment':1,'comparable_implementation':protocol['implementation_revision'],'training_stop_utc':window['training_stop_utc'],'expected_boot_id':boot}
    out=R/'runs'/attempt;out.mkdir(exist_ok=False);save(out/'request.json',req);save(out/'prelaunch.json',dict(samples=samples,wait_s=time.time()-waiting_start))
    start_qpc=qpc();start_utc=utc();append(ledger,dict(event='ATTEMPT_START',status='RUNNING',slot_id=slot['slot_id'],attempt_id=attempt,attempt_number=attempt_number,start_utc=start_utc,start_qpc=start_qpc))
    status('RUNNING',training_started=True,active_attempt=attempt,valid_runs=len(valid),next_slot=None)
    cmd=['wsl','-d',DISTRO,'-u','root','--exec','/opt/s09/venv/bin/python','-u',G+'/src/guest_worker.py','run',GH+'/runs/'+attempt+'/request.json']
    with (out/'host_runner.log').open('wb') as f:
     child=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT)
     save(out/'host_process.json',dict(pid=child.pid,created_time=psutil.Process(child.pid).create_time(),args=cmd,started_utc=start_utc))
     while child.poll() is None:
      if qpc()>=stop_qpc:save(R/'control/BUDGET_STOP',dict(reason='Host QPC training deadline',utc=utc()))
      time.sleep(3)
      # Guest guard honors the same fixed deadline, user latch and heartbeat.
     rc=child.wait()
    time.sleep(2)
    try:
     assert rc==0,('guest wrapper failed',rc)
     core=read(out/'core_validation.json');assert core['status']=='PASS',core
     result=read(out/'result.json');assert result['boot_id']==boot
     timing=monitor.validate(result['measurement_start_raw_ns'],result['measurement_end_raw_ns'],boot)
     # Any observable affinity violation invalidates the process-placement contract.
     violations=[]
     for line in (out/'guest_resources.jsonl').read_text().splitlines():
      sample=json.loads(line)
      if not result['measurement_start_raw_ns']<=sample['raw_ns']<=result['measurement_end_raw_ns']:continue
      for who in ('training','writer'):
       a=(sample.get(who) or {}).get('affinity')
       if a and (a['violations'] or a['unreadable']):violations.append(a)
     assert not violations,'Measured affinity violation'
     validation=dict(status='PASS',timing=timing,core=core,source_sha256=p_sha,utc=utc());save(out/'host_validation.json',validation)
     final=guest(['/opt/s09/venv/bin/python',G+'/src/guest_worker.py','finalize',attempt],capture_output=True,text=True)
     assert final.returncode==0,final.stdout+final.stderr
     record=dict(event='ATTEMPT_END',status='VALID',slot_id=slot['slot_id'],attempt_id=attempt,attempt_number=attempt_number,start_utc=start_utc,end_utc=utc(),host_elapsed_s=(qpc()-start_qpc)/F,result_sha256=sha(out/'result.json'),elapsed_s=result['elapsed_s'],session_id=session,boot_id=boot)
     append(ledger,record);valid[slot['slot_id']]=record;status('BETWEEN_RUNS',active_attempt=None,valid_runs=len(valid));break
    except Exception as e:
     failed=dict(event='ATTEMPT_END',status='FAILED',slot_id=slot['slot_id'],attempt_id=attempt,attempt_number=attempt_number,error=repr(e),traceback=traceback.format_exc(),start_utc=start_utc,end_utc=utc(),host_elapsed_s=(qpc()-start_qpc)/F)
     append(ledger,failed);save(out/'failure.json',failed)
     if pause():status('USER_PAUSED',active_attempt=None);return
     signature=repr(e);failed_signatures.append(signature)
     action='retry' if attempt_number<3 else 'missing'
     save(out/'recovery_decision.json',dict(failed_attempt=attempt,objective_evidence=str(out/'failure.json'),recovery_action='Fresh isolated process after verified previous wrapper exit; retain failure evidence',expected_resolution='Transient process/runtime failure clears on fresh launch; common failures require agent repair',action=action))
     if len(failed_signatures)>=2 and failed_signatures[-1]==failed_signatures[-2]:raise RuntimeError('Repeated technical failure; inspect and repair before dispatching further: '+signature)
   else:status('BETWEEN_RUNS',active_attempt=None,missing_slot=slot['slot_id'])
   if slot['schedule_index']==7:
    summary={}
    for condition in ('OA','O0'):
     selected=[]
     for q in protocol['fixed_queue'][:8]:
      if q['condition']==condition and q['slot_id'] in valid:
       v=read(R/'runs'/valid[q['slot_id']]['attempt_id']/'result.json');selected.append(dict(attempt_id=v['attempt_id'],elapsed_s=v['elapsed_s'],fetch_s=sum(x['fetch_s'] for x in v['update_records'])))
     values=[x['elapsed_s'] for x in selected]
     summary[condition]=dict(runs=selected,n=len(values),mean_s=statistics.mean(values) if values else None,range_s=max(values)-min(values) if values else None,sd_s=statistics.stdev(values) if len(values)>1 else None)
    save(R/'evidence/initial_diagnostic_summary.json',dict(status='DESCRIPTIVE_ONLY_CONTINUE_MAIN',utc=utc(),summary=summary,inclusion_rule='No performance or variance gate; all technically valid observations retained'))
  status('COLLECTION_FINISHED',active_attempt=None,valid_runs=len(valid),collection_ended_utc=utc())
 except BaseException as e:
  status('NEEDS_REPAIR',error=repr(e),traceback=traceback.format_exc());raise
 finally:
  monitor.close();heartbeat_stop.set();hb.join(10);ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
  save(R/'host/controller_exit.json',dict(utc=utc(),pid=os.getpid(),monitor_errors=monitor.errors,state=state['status'],monitor_children=[dict(pid=p.pid,returncode=p.poll()) for p in monitor.children]))
  lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1);lock.close()
if __name__=='__main__':main()
