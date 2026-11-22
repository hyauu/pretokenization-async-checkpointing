"""Continue preparation through verified protocol freeze and real controller launch."""
import datetime,hashlib,json,os,pathlib,shutil,subprocess,sys,time,traceback
R=pathlib.Path(__file__).resolve().parents[1];DISTRO='S09Replication-20261027';G='/opt/s09_replication_20261027';GH='/mnt/c/Users/anonymous/Documents/ChatGPT/S09Replication_2026-10-27'
def save(p,v):
 p=R/p;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,indent=2));os.replace(t,p)
def phase(name,**kw):
 save('setup/pipeline_status.json',dict(status=name,pid=os.getpid(),utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**kw));print(name,kw,flush=True)
def run(args,label,env=None):
 with (R/'setup'/f'{label}.log').open('wb') as f:
  p=subprocess.Popen(args,stdout=f,stderr=subprocess.STDOUT,env=env);save('setup/pipeline_child.json',dict(pid=p.pid,args=args,label=label));rc=p.wait()
 if rc:raise RuntimeError(f'{label} failed {rc}; inspect setup/{label}.log')
def guest(args,label):run(['wsl','-d',DISTRO,'-u','root','--exec',*args],label)
try:
 phase('WAITING_FOR_C_IMPORT')
 while True:
  if (R/'control/USER_PAUSED').exists():phase('USER_PAUSED');sys.exit(0)
  current=json.loads((R/'setup/status.json').read_text())
  if current['status']=='ENVIRONMENT_IMPORTED':break
  if current['status']=='SETUP_FAILED':raise RuntimeError(current)
  time.sleep(10)
 phase('COPYING_ALPACA_AND_REMAINING_PINNED_PACKAGE')
 extra=GH+'/setup/remaining_assets.tar'
 members=['opt/s09/data/alpaca.json','opt/s09/data/alpaca_README.md','opt/s09/data/cache_labels.npy','opt/s09/data/cache_masks.npy','opt/s09/data/cache_offsets.npy','opt/s09/data/cache_tokens.npy','opt/s09/upstream']
 run(['wsl','-d','S09Pilot-20261020','-u','root','--exec','tar','--numeric-owner','-cpf',extra,'-C','/',*members],'remaining_export')
 guest(['tar','-xpf',extra,'-C','/'],'remaining_import')
 phase('INSTALLING_ROUND_FILES')
 code="import pathlib,shutil; h=pathlib.Path('"+GH+"'); g=pathlib.Path('"+G+"'); g.mkdir(exist_ok=True); shutil.copytree(h/'src',g/'src',dirs_exist_ok=True); (g/'runs').mkdir(exist_ok=True); (g/'checkpoints').mkdir(exist_ok=True)"
 guest(['/usr/bin/python3','-c',code],'install_round')
 phase('VERIFYING_C_ASSETS')
 guest(['env','CUDA_VISIBLE_DEVICES=','PYTHONPATH=/opt/s09/upstream','/opt/s09/venv/bin/python',GH+'/setup/verify_environment.py'],'verify_assets')
 phase('FREEZING_PROTOCOL')
 run([sys.executable,'-X','utf8',str(R/'host/build_protocol.py')],'freeze_protocol')
 files={str(p.relative_to(R)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for folder in ('host','src','setup') for p in sorted((R/folder).glob('*.py'))}
 save('execution_protocol.json',dict(round_id='s09_replication_20261027',source_sha256=files,protocol_sha256=hashlib.sha256((R/'protocol.json').read_bytes()).hexdigest(),window=json.loads((R/'execution_window.json').read_text(encoding='utf-8-sig')),created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
 guest(['/usr/bin/python3','-c',"import shutil;shutil.copy2('"+GH+"/protocol.json','"+G+"/protocol.json')"],'install_protocol')
 guest(['env','CUDA_VISIBLE_DEVICES=','PYTHONPATH='+G+'/src','/opt/s09/venv/bin/python',GH+'/setup/check_contract.py'],'check_contract')
 phase('LAUNCHING_CONTROLLER')
 if (R/'control/USER_PAUSED').exists():phase('USER_PAUSED');sys.exit(0)
 run([sys.executable,'-X','utf8','-u',str(R/'host/controller.py')],'controller')
 phase('CONTROLLER_EXITED',state=json.loads((R/'state.json').read_text())['status'])
except BaseException as e:
 if isinstance(e,SystemExit):raise
 phase('NEEDS_REPAIR',error=repr(e),traceback=traceback.format_exc());traceback.print_exc();sys.exit(1)
