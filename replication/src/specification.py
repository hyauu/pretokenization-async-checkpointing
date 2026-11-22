"""Frozen queue and workload contract for the independent controlled replication."""
import math
ROUND='s09_replication_20261027'
KEYS=('slot_id','group_id','family','workload','block','repeat_block','condition','interval','save_schedule','placement','training_cpus','writer_cpus','layout','half_pool','training_pool','position','schedule_index','manifest','anchor','updates')
def validate_request(r,p,m):
 assert r['round_id']==p['round_id']==ROUND
 frozen=next(x for x in p['fixed_queue'] if x['slot_id']==r['slot_id'])
 assert all(r[k]==frozen[k] for k in KEYS)
 assert r['action']=='run' and r['attempt_id'].startswith(r['slot_id']+'_a')
 assert r['attempt_number'] in (1,2,3)
 assert r['session_id'] and r['measurement_segment']>=1 and r['input_dose']==1
 assert r['comparable_implementation']==p['implementation_revision']
 assert m['microbatch']==2 and m['accumulation']==8 and m['warmup_updates']==3
 assert m['measurement_updates']==r['updates']
 assert m['manifest_sha256']==p['manifests'][f"{r['workload']}_{r['block']}"]['manifest_sha256']
 saving=r['condition'].endswith('A')
 assert bool(r['interval'])==saving and r['interval']>=0
 assert r['save_schedule'] in (('overlap','serialize') if saving else ('none',))
 assert len(r['training_cpus']) in (8,16) and len(set(r['training_cpus']))==len(r['training_cpus'])
 assert all(0<=x<16 for x in r['training_cpus']+r['writer_cpus'])
 assert (len(r['writer_cpus']) in (8,16)) if saving else (r['writer_cpus']==[] and r['placement']=='absent')
 return frozen
def validate_result(r,p,m,v):
 validate_request(r,p,m)
 n=r['updates'];saving=r['condition'].endswith('A')
 assert v['completed_updates']==len(v['update_records'])==n
 assert v['effective_supervised_tokens']==sum(x['tokens'] for x in v['update_records'])==m['effective_supervised_tokens']
 assert all(math.isfinite(x['loss']) for x in v['update_records'])
 assert v['final_global_step']==n+3 and v['final_consumed_cursor']==8*(n+3)
 assert v['initial_checkpoint_commits']==int(saving)
 assert v['checkpoint_commits']==(n//r['interval'] if saving else 0)
 expected=p['anchors'][f"{r['workload']}_{r['block']}"]
 assert all(expected[k]==x for k,x in v['measurement_initial_state'].items())
 assert abs(v['elapsed_s']-(v['measurement_end_raw_ns']-v['measurement_start_raw_ns'])/1e9)<1e-8
 assert [x['measurement_update'] for x in v['update_records']]==list(range(1,n+1))
 for key in ('placement_pre','placement_post'):
  for who in ('training','writer'):
   t=v[key][who]
   if t:assert t['readable'] and not t['violations'] and not t['unreadable']
 commits=[x for x in v['save_events'] if x['event']=='commit']
 assert [x['step'] for x in commits]==([3]+list(range(3+r['interval'],n+4,r['interval'])) if saving else [])
 if not saving:assert not v['save_events'] and v['placement_info']['writer'] is None
 if r['save_schedule']=='serialize':
  for record in v['update_records']:
   if record['measurement_update']%r['interval']==0:
    assert record['post_save_drain_start_raw_s']<=record['post_save_drain_end_raw_s']
    commit=next(x for x in commits if x['step']==record['global_step'])
    assert commit['end']<=record['post_save_drain_end_raw_s']
 return True
