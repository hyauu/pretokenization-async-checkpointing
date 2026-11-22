"""Archive only this round's validated final checkpoints, then read back every member."""
import datetime,hashlib,json,os,pathlib,tarfile,time
G=pathlib.Path('/opt/s09_replication_20261027');H=pathlib.Path('/mnt/c/Users/anonymous/Documents/ChatGPT/S09Replication_2026-10-27')
D=pathlib.Path('/mnt/d/Documents/ChatGPT/S09Replication_2026-10-27')
def save(p,v):
 t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,indent=2));os.replace(t,p)
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def main():
 for p in pathlib.Path('/proc').glob('[0-9]*/cmdline'):
  try:cmd=p.read_bytes().decode(errors='ignore')
  except (FileNotFoundError,ProcessLookupError,PermissionError):continue
  assert not ('followup_train.py' in cmd and str(G) in cmd),'A training process remains'
 D.mkdir(parents=True,exist_ok=True)
 target=D/'final_checkpoints.tar';partial=D/'final_checkpoints.tar.partial'
 if target.exists() or partial.exists():raise RuntimeError('Archive destination already exists; reconstruct/verify before resuming')
 rows=[]
 ledger=[json.loads(x) for x in (H/'evidence/gpu_ledger.jsonl').read_text().splitlines()]
 for e in ledger:
  if e.get('status')!='VALID':continue
  r=json.loads((H/'runs'/e['attempt_id']/'retention.json').read_text())['retained_final']
  if not r:continue
  folder=pathlib.Path(r['path']);assert folder.resolve().is_relative_to((G/'checkpoints').resolve())
  for file in r['files']:
   path=folder/file['name'];assert path.is_file() and not path.is_symlink() and path.stat().st_size==file['bytes']
   rows.append(dict(source=str(path),member=str(path.relative_to(G)),bytes=file['bytes'],sha256=file['sha256']))
 save(H/'delivery/archive_progress.json',dict(status='COPYING',utc=now(),members=len(rows),bytes=sum(r['bytes'] for r in rows),destination=str(target)))
 with tarfile.open(partial,'w') as tar:
  for i,r in enumerate(rows):
   tar.add(r['source'],arcname=r['member'],recursive=False)
   save(H/'delivery/archive_progress.json',dict(status='COPYING',utc=now(),completed_members=i+1,members=len(rows),destination=str(target)))
 with partial.open('rb') as f:os.fsync(f.fileno())
 os.replace(partial,target)
 expected={r['member']:r for r in rows};verified=[]
 with tarfile.open(target,'r:') as tar:
  for member in tar:
   if not member.isfile():continue
   assert member.name in expected,member.name;r=expected[member.name];h=hashlib.sha256()
   f=tar.extractfile(member)
   for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
   assert member.size==r['bytes'] and h.hexdigest()==r['sha256'],member.name
   verified.append(member.name);save(H/'delivery/archive_progress.json',dict(status='VERIFYING',utc=now(),verified_members=len(verified),members=len(rows)))
 assert set(verified)==set(expected) and len(verified)==len(expected)
 receipt=dict(status='PASS',completed_utc=now(),archive_windows_path='D:\\Documents\\ChatGPT\\S09Replication_2026-10-27\\final_checkpoints.tar',archive_bytes=target.stat().st_size,members=rows,verified_members=len(verified),verification='Full readback SHA256 of every archived file against post-run checkpoint inventory; no extraction required')
 save(D/'checkpoint_archive_verification.json',receipt);save(H/'delivery/checkpoint_archive_verification.json',receipt)
 save(H/'delivery/archive_progress.json',dict(status='PASS',utc=now(),verified_members=len(verified)))
 print(json.dumps({'status':'PASS','members':len(verified),'archive_bytes':target.stat().st_size}))
if __name__=='__main__':main()
