"""Check compact methods evidence using metadata only; never execute excerpts."""
from pathlib import Path
from collections import Counter
import hashlib,json
root=Path(__file__).resolve().parents[1]/'evidence/methods'
manifest=json.loads((root/'MEMBER_SHA256.json').read_text(encoding='utf-8'))
for name,v in manifest.items():
 p=root/name
 assert p.stat().st_size==v['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==v['sha256'],name
assert {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and p.name!='MEMBER_SHA256.json'}==set(manifest)
expected={('model','torch.bfloat16'):310,('exp_avg','torch.bfloat16'):310,('exp_avg_sq','torch.bfloat16'):310,('step','torch.float32'):310}
reported=json.loads((root/'schema/dtype_counts.json').read_text())['signatures']
for name,key in [('capture_signature.json','capture'),('restored_continued_signature.json','restored_continuation')]:
 records=json.loads((root/'schema'/name).read_text())['tensors'];counts=Counter()
 for t in records:
  field='model' if t['path'].startswith('/model/') else t['path'].rsplit('/',1)[-1]
  counts[field,t['dtype']]+=1
 assert len(records)==1240 and dict(counts)==expected,(name,counts)
 assert {(x['field'],x['dtype']):x['count'] for x in reported[key]['counts']}==dict(counts)
print('PASS: methods member hashes and both 1,240-tensor dtype counts.')
