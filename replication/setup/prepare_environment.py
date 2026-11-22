"""Build an independent C-backed WSL runtime from selected frozen local assets."""
import datetime,json,os,pathlib,subprocess,sys,time,traceback
R=pathlib.Path(__file__).resolve().parents[1]
OLD='S09Pilot-20261020'; NEW='S09Replication-20261027'
def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(name,obj):
 p=R/name;p.parent.mkdir(parents=True,exist_ok=True)
 t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(obj,indent=2),encoding='utf-8');os.replace(t,p)
def phase(s,**kw):
 save('setup/status.json',dict(status=s,updated_utc=now(),pid=os.getpid(),**kw))
 print(now(),s,kw,flush=True)
def run(args,label,allowed=(0,)):
 with (R/'setup'/f'{label}.log').open('wb') as f:
  p=subprocess.Popen(args,stdout=f,stderr=subprocess.STDOUT)
  save('setup/active_child.json',dict(pid=p.pid,args=args,label=label,started_utc=now()))
  rc=p.wait()
 if rc not in allowed:raise RuntimeError(f'{label} exit {rc}; inspect setup/{label}.log')
 return rc
try:
 phase('EXPORTING_SELECTED_RUNTIME')
 tar=R/'setup/runtime.tar'
 if tar.exists():raise RuntimeError('Refusing to overwrite existing runtime.tar')
 guest_tar='/mnt/c/Users/anonymous/Documents/ChatGPT/S09Replication_2026-10-27/setup/runtime.tar'
 members=['bin','bin.usr-is-merged','boot','etc','init','lib','lib.usr-is-merged','lib64','sbin','sbin.usr-is-merged','usr','var',
          'opt/s09/venv','opt/s09/upstream/torchtune','opt/s09/data/model',
          'opt/s09_followup_20261021/raw_v2/anchors','opt/s09_followup_20261021/raw_v2/data',
          'opt/s09_followup_20261021/raw_v2/evidence/manifests']
 excludes=['var/log','var/cache','var/tmp','var/lib/snapd','var/lib/systemd/random-seed','etc/machine-id']
 args=['wsl','-d',OLD,'-u','root','--exec','tar','--numeric-owner','--one-file-system',
       *['--exclude='+v for v in excludes],'-cpf',guest_tar,'-C','/',*members]
 rc=run(args,'export',allowed=(0,1))
 if rc:
  log=(R/'setup/export.log').read_text(errors='replace')
  if 'Cannot' in log or 'Error' in log or 'No such' in log:raise RuntimeError('Export reported an actual read error')
 phase('IMPORTING_C_RUNTIME',tar_bytes=tar.stat().st_size)
 run(['wsl','--import',NEW,str(R/'wsl'),str(tar),'--version','2'],'import')
 phase('CONFIGURING_C_RUNTIME')
 code="from pathlib import Path; Path('/etc/wsl.conf').write_text('[boot]\\nsystemd=false\\n[user]\\ndefault=root\\n'); [Path(p).mkdir(parents=True,exist_ok=True) for p in ['/tmp','/root','/home','/mnt/c','/dev','/proc','/sys','/run','/opt/s09_replication_20261027']]; Path('/tmp').chmod(0o1777)"
 run(['wsl','-d',NEW,'-u','root','--exec','/usr/bin/python3','-c',code],'configure')
 run(['wsl','--terminate',NEW],'restart_config')
 phase('ENVIRONMENT_IMPORTED',tar_bytes=tar.stat().st_size,training_started=False)
except BaseException as e:
 phase('SETUP_FAILED',error=repr(e));traceback.print_exc();sys.exit(1)
