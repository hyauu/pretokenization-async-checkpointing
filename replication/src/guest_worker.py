"""Own one isolated training process group, lightweight telemetry, bounded cleanup."""
import argparse,datetime,hashlib,json,os,pathlib,shutil,signal,subprocess,sys,threading,time,traceback
import psutil
from timing import raw_ns
from placement import identity,process_sample
G=pathlib.Path('/opt/s09_replication_20261027')
H=pathlib.Path('/mnt/c/Users/anonymous/Documents/ChatGPT/S09Replication_2026-10-27')
def save(p,v):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,indent=2));os.replace(t,p)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def group_members(pgid):
 rows=[]
 for p in psutil.process_iter(['pid','status','cmdline']):
  try:
   if os.getpgid(p.pid)==pgid and p.info['status']!=psutil.STATUS_ZOMBIE:rows.append(identity(p.pid))
  except (ProcessLookupError,psutil.NoSuchProcess,PermissionError):pass
 return rows
def stop_group(who):
 members=group_members(who['pgid'])
 assert all(x['start_ticks']>=who['start_ticks'] and x['boot_id']==who['boot_id'] for x in members)
 if members:
  os.killpg(who['pgid'],signal.SIGTERM)
  for _ in range(20):
   if not group_members(who['pgid']):return
   time.sleep(.25)
  if group_members(who['pgid']):os.killpg(who['pgid'],signal.SIGKILL)
def telemetry(out,who,request,done):
 with (out/'guest_resources.jsonl').open('a',buffering=1) as f:
  count=0
  while not done.is_set():
   row={'raw_ns':raw_ns(),'memory':psutil.virtual_memory()._asdict(),'swap':psutil.swap_memory()._asdict(),'cpu_times':psutil.cpu_times()._asdict(),'disk':{k:v._asdict() for k,v in (psutil.disk_io_counters(perdisk=True) or {}).items()}}
   ready=out/'placement_ready.json'
   try:
    if ready.exists():
     place=json.loads(ready.read_text());row['training']=process_sample(place['training'],request['training_cpus'],count%2==0);row['writer']=process_sample(place['writer'],request['writer_cpus'],count%2==0)
   except Exception as e:row['sample_error']=repr(e)
   f.write(json.dumps(row)+'\n');count+=1;done.wait(5)
def transfer(out):
 dst=H/'runs'/out.name;dst.mkdir(parents=True,exist_ok=True)
 for p in out.iterdir():
  if p.is_file():shutil.copy2(p,dst/p.name)
def run(request_path):
 r=json.loads(pathlib.Path(request_path).read_text(encoding='utf-8-sig'))
 p=json.loads((G/'protocol.json').read_text());m=json.loads(pathlib.Path(r['manifest']).read_text())
 from specification import validate_request,validate_result
 validate_request(r,p,m)
 assert pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()==r['expected_boot_id']
 out=G/'runs'/r['attempt_id'];out.mkdir(parents=True,exist_ok=False)
 save(out/'request.json',r)
 env=os.environ.copy();env.update(S09_ROOT=str(G),S09_WORKLOAD=r['workload'],S09_DATA_ROOT='/opt/s09/data' if r['workload']=='A' else '/opt/s09_followup_20261021/raw_v2/data/ultrachat',PYTHONPATH=str(G/'src')+':/opt/s09/upstream',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8')
 env.pop('CUDA_VISIBLE_DEVICES',None)
 cmd=[sys.executable,'-u',str(G/'src/placement_launcher.py'),'--request',str(out/'request.json')]
 began=raw_ns();reason=None;done=threading.Event()
 with (out/'stdout.log').open('wb') as log:
  child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
  who=identity(child.pid);save(out/'group.json',{'owner':identity(os.getpid()),'training':who,'started_raw_ns':began,'command':cmd})
  sampler=threading.Thread(target=telemetry,args=(out,who,r,done),daemon=True);sampler.start()
  stop_at=datetime.datetime.fromisoformat(r['training_stop_utc'].replace('Z','+00:00')).timestamp()
  try:
   while child.poll() is None:
    if (H/'control/USER_PAUSED').exists():reason='USER_PAUSED'
    elif (H/'control/BUDGET_STOP').exists():reason='BUDGET_STOP'
    elif time.time()>=stop_at:reason='BUDGET_STOP'
    elif not (H/'host/heartbeat.json').exists():reason='HOST_HEARTBEAT_MISSING'
    elif time.time()-(H/'host/heartbeat.json').stat().st_mtime>120:reason='HOST_HEARTBEAT_STALE'
    elif shutil.disk_usage(G).free<8*1024**3:reason='DISK_CRITICAL'
    if reason:break
    time.sleep(2)
  except BaseException as e:reason='GUARD_ERROR: '+repr(e)
  finally:
   stop_group(who);rc=child.wait();done.set();sampler.join(15)
 ended=raw_ns();members=group_members(who['pgid'])
 accounting=dict(attempt_id=r['attempt_id'],started_raw_ns=began,closed_raw_ns=ended,elapsed_process_group_s=(ended-began)/1e9,return_code=rc,stop_reason=reason,remaining_members=members,boot_id=who['boot_id'],owner=who)
 save(out/'accounting.json',accounting)
 try:
  assert rc==0 and not reason and not members,accounting
  v=json.loads((out/'result.json').read_text());validate_result(r,p,m,v)
  inventory=[]
  for event in v['save_events']:
   if event['event']!='commit':continue
   folder=pathlib.Path(event['path']);assert folder.resolve().is_relative_to((G/'checkpoints'/r['attempt_id']).resolve())
   flag=json.loads((folder/'COMPLETE').read_text());assert flag['global_step']==event['step'] and flag['protocol_id']==r['protocol_sha256']
   files=[{'name':f.name,'bytes':f.stat().st_size,'sha256':sha(f)} for f in sorted(folder.iterdir()) if f.is_file()]
   assert {f['name'] for f in files}=={'.metadata','__0_0.distcp','COMPLETE'}
   inventory.append({'path':str(folder),'step':event['step'],'initial':event['initial'],'files':files,'bytes':sum(f['bytes'] for f in files)})
  assert len(inventory)==v['checkpoint_commits']+v['initial_checkpoint_commits']
  save(out/'checkpoint_inventory.json',{'status':'PASS','checkpoints':inventory})
  save(out/'core_validation.json',{'status':'PASS','training_group_closed':True,'result_sha256':sha(out/'result.json')})
  transfer(out);print(json.dumps({'status':'CORE_VALIDATED','attempt_id':r['attempt_id'],'elapsed_s':v['elapsed_s']}),flush=True)
 except Exception as e:
  save(out/'core_validation.json',{'status':'FAIL','error':repr(e),'traceback':traceback.format_exc()});transfer(out);raise
def finalize(attempt):
 out=G/'runs'/attempt
 receipt=json.loads((H/'runs'/attempt/'host_validation.json').read_text())
 assert receipt['status']=='PASS'
 inv=json.loads((out/'checkpoint_inventory.json').read_text())['checkpoints']
 base=(G/'checkpoints'/attempt).resolve();removed=[]
 for row in inv[:-1]:
  folder=pathlib.Path(row['path']);resolved=folder.resolve()
  assert resolved.is_relative_to(base) and resolved!=base and not folder.is_symlink()
  assert all((folder/f['name']).stat().st_size==f['bytes'] for f in row['files'])
  shutil.rmtree(folder);removed.append(str(folder))
 save(out/'retention.json',{'removed_verified_intermediate':removed,'retained_final':inv[-1] if inv else None})
 transfer(out);print(json.dumps({'status':'RETAINED','attempt_id':attempt,'removed':len(removed)}))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('action',choices=['run','finalize']);ap.add_argument('argument');a=ap.parse_args()
 try:(run if a.action=='run' else finalize)(a.argument)
 except BaseException:traceback.print_exc();sys.exit(1)
