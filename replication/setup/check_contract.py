import copy,json,pathlib,sys
G=pathlib.Path('/opt/s09_replication_20261027');sys.path.insert(0,str(G/'src'))
from specification import validate_request
p=json.loads((G/'protocol.json').read_text());n=bad=0
for slot in p['fixed_queue']:
 m=json.loads(pathlib.Path(slot['manifest']).read_text())
 for attempt in (1,2,3):
  r=dict(slot,round_id=p['round_id'],action='run',attempt_id=slot['slot_id']+f'_a{attempt}',attempt_number=attempt,session_id='contract_test',measurement_segment=1,comparable_implementation=p['implementation_revision'])
  validate_request(r,p,m);n+=1
  for key,value in [('updates',r['updates']+1),('training_cpus',[0]),('condition','P0' if r['condition']!='P0' else 'OA')]:
   b=copy.deepcopy(r);b[key]=value
   try:validate_request(b,p,m)
   except (AssertionError,StopIteration):bad+=1
   else:raise AssertionError(('Negative request accepted',key))
print(json.dumps(dict(status='PASS',legal_requests=n,rejected_mutations=bad,torch_imported='torch' in sys.modules)))
assert 'torch' not in sys.modules
