"""Set training affinity before torch import or training thread creation."""
import argparse,json,os,sys
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--request',required=True);args=ap.parse_args()
r=json.loads(Path(args.request).read_text());mask=r['training_cpus']
p=json.loads(Path(r['protocol']).read_text())
frozen=next(x for x in p['fixed_queue'] if x['slot_id']==r['slot_id'])
assert mask==frozen['training_cpus'] and len(mask) in (8,16)
assert set(mask)<=os.sched_getaffinity(0)
assert os.environ.get('CUDA_VISIBLE_DEVICES')!='','CPU-only test environment leaked into GPU training'
os.sched_setaffinity(0,mask)
assert sorted(os.sched_getaffinity(0))==mask
os.execv(sys.executable,[sys.executable,'-u',str(Path(__file__).with_name('followup_train.py')),'--request',args.request])
