"""Wait for collection closure, analyze, archive and verify; agent then reviews figures."""
import datetime,hashlib,json,msvcrt,os,pathlib,subprocess,sys,time,traceback,zipfile,shutil
R=pathlib.Path(__file__).resolve().parents[1]
def save(p,v):
 p=R/p;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,indent=2));os.replace(t,p)
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def phase(s,**kw):save('host/finalizer_status.json',dict(status=s,pid=os.getpid(),utc=now(),**kw));print(s,kw,flush=True)
def run(args,label):
 with (R/'host'/f'{label}.log').open('wb') as f:
  p=subprocess.Popen(args,stdout=f,stderr=subprocess.STDOUT);save('host/finalizer_child.json',dict(pid=p.pid,args=args));rc=p.wait()
 if rc:raise RuntimeError(f'{label} failed with {rc}; inspect host/{label}.log')
def package():
 target=R/'delivery/S09_Replication_Evidence_Source.zip';files=[]
 for folder in ['src','host','setup','evidence','runs','delivery']:
  for f in (R/folder).rglob('*'):
   if not f.is_file() or '__pycache__' in f.parts or f.suffix in ('.tar','.zip','.pyc','.partial','.lock') or f.stat().st_size>300*1024**2:continue
   if f.name in ('package_verification.json','archive_progress.json'):continue
   files.append(f)
 for name in ('RUNBOOK.md','PLAN.md','protocol.json','execution_protocol.json','execution_window.json','session.json','state.json'):files.append(R/name)
 manifest={str(f.relative_to(R)).replace('\\','/'):hashlib.sha256(f.read_bytes()).hexdigest() for f in files if f.exists()}
 with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for name in manifest:z.write(R/name,name)
  z.writestr('MEMBER_SHA256.json',json.dumps(manifest,indent=2))
 with zipfile.ZipFile(target) as z:
  assert set(z.namelist())==set(manifest)|{'MEMBER_SHA256.json'}
  for name,h in manifest.items():assert hashlib.sha256(z.read(name)).hexdigest()==h,name
 receipt=dict(status='PASS',members=len(manifest),zip_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),bytes=target.stat().st_size,utc=now())
 save('delivery/package_verification.json',receipt)
 archive=pathlib.Path(r'D:\Documents\ChatGPT\S09Replication_2026-10-27');archive.mkdir(parents=True,exist_ok=True)
 for f in (R/'delivery').iterdir():
  if f.is_file():
   dst=archive/f.name;shutil.copy2(f,dst);assert hashlib.sha256(dst.read_bytes()).digest()==hashlib.sha256(f.read_bytes()).digest()
 return receipt
def main():
 lock=(R/'host/finalizer.lock').open('a+b');lock.seek(0);lock.write(b'1');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
 phase('WAITING_FOR_COLLECTION')
 while True:
  s=json.loads((R/'state.json').read_text(encoding='utf-8-sig'))
  if s['status'] in ('COLLECTION_FINISHED','BUDGET_COLLECTION_CLOSED'):break
  if s['status']=='DELIVERED':phase('ALREADY_DELIVERED');return
  time.sleep(15)
 while not (R/'host/controller_exit.json').exists():time.sleep(2)
 phase('ANALYZING');run([sys.executable,'-X','utf8',str(R/'host/analyze.py')],'analysis')
 phase('ARCHIVING_FINAL_CHECKPOINTS')
 run(['wsl','-d','S09Replication-20261027','-u','root','--exec','/opt/s09/venv/bin/python','/mnt/c/Users/anonymous/Documents/ChatGPT/S09Replication_2026-10-27/setup/archive_checkpoints.py'],'archive')
 phase('PACKAGING');receipt=package()
 phase('READY_FOR_VISUAL_REVIEW',package=receipt)
 s.update(status='READY_FOR_VISUAL_REVIEW',updated_utc=now());save('state.json',s)
if __name__=='__main__':
 try:main()
 except BaseException as e:phase('FINALIZATION_NEEDS_REPAIR',error=repr(e),traceback=traceback.format_exc());raise
