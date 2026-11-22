"""Single offline reconstruction entry point. Run in a fresh extraction."""
from pathlib import Path
import subprocess,sys,hashlib,json
root=Path(__file__).resolve().parents[1]
for script,args in [('verify_manifest.py',[]),('verify_methods.py',[]),('reproduce.py',['--skip-figure']),('reproduce_icict.py',[])]:
 subprocess.run([sys.executable,'-X','utf8',str(root/'analysis'/script),*args],cwd=root,check=True)
for name,digest in json.loads((root/'evidence/numeric_baseline.json').read_text()).items():
 assert hashlib.sha256((root/'data/icict'/name).read_bytes()).hexdigest()==digest,('numerical baseline',name)
print('PASS: complete offline reconstruction; all eight numerical CSVs match v0.4 exactly.')
