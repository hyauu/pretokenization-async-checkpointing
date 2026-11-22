"""One-time CPU asset verification in the new C-backed guest only."""
import hashlib,json,pathlib,subprocess,sys,os,datetime
G=pathlib.Path('/opt/s09_replication_20261027')
H=pathlib.Path('/mnt/c/Users/anonymous/Documents/ChatGPT/S09Replication_2026-10-27')
old=json.loads((H/'evidence/old_initial_protocol.json').read_text(encoding='utf-8-sig'))
allocation=json.loads((H/'evidence/old_allocation_protocol.json').read_text(encoding='utf-8-sig'))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
checks=[]
def verify(path,expected):
 p=pathlib.Path(path);actual=sha(p);assert actual==expected,(str(p),actual,expected)
 checks.append({'path':str(p),'bytes':p.stat().st_size,'sha256':actual})
for item in json.loads((H/'evidence/old_asset_identity.json').read_text(encoding='utf-8-sig'))['assets_checked']:
 verify(item['path'],item['sha256'])
assets=old['data_manifest']['alpaca_asset_manifest']
verify('/opt/s09/data/alpaca.json',assets['source_sha256'])
for name,v in assets['cache'].items():verify('/opt/s09/data/'+name,v['sha256'])
anchors={};manifests={}
for w in ('A','U'):
 for b in (1,2,3):
  key=f'{w}_{b}';path=pathlib.Path(f'/opt/s09_followup_20261021/raw_v2/anchors/{key}.pt')
  meta=json.loads(path.with_suffix('.json').read_text());verify(path,meta['file_sha256'])
  expected=old['anchors'][key]
  assert all(meta[k]==v for k,v in expected.items()),key
  mf=json.loads(pathlib.Path(f'/opt/s09_followup_20261021/raw_v2/evidence/manifests/{w}_block_{b}.json').read_text())
  assert mf['manifest_sha256']==meta['manifest_sha256']
  body={k:v for k,v in mf.items() if k!='manifest_sha256'}
  assert hashlib.sha256(json.dumps(body,sort_keys=True,separators=(',',':')).encode()).hexdigest()==mf['manifest_sha256']
  anchors[key]=meta;manifests[key]={k:mf[k] for k in ('manifest_sha256','measurement_updates','effective_supervised_tokens','seed')}
for name in ('checkpoint.py','train_core.py','timing.py'):
 verify(G/'src'/name,allocation['source_sha256'][name])
filesystem=subprocess.check_output(['findmnt','-n','-o','FSTYPE,SOURCE','-T',str(G)],text=True).strip()
assert filesystem.startswith('ext4 '),filesystem
assert set(range(16))<=os.sched_getaffinity(0)
import torch,torchtune
assert torch.__version__=='2.14.0+cu130',torch.__version__
assert torch.version.git_version=='08187d9e0fba026dc8217405802ab5381dc88d90',torch.version.git_version
metadata=dict(status='PASS',checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),assets_checked=checks,anchors=anchors,manifests=manifests,filesystem=filesystem,torch=torch.__version__,torch_git=torch.version.git_version,python=sys.version,torchtune_path=torchtune.__file__,cpu_mask=sorted(os.sched_getaffinity(0)),boot_id=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip())
(H/'evidence/asset_metadata.json').write_text(json.dumps(metadata,indent=2))
print(json.dumps({'status':'PASS','assets':len(checks),'filesystem':filesystem,'torch':torch.__version__}))
