import pathlib,json,hashlib,datetime,collections
R=pathlib.Path(__file__).resolve().parents[1]
G='/opt/s09_replication_20261027'; AS='/opt/s09_followup_20261021/raw_v2'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def build():
 meta=json.loads((R/'evidence/asset_metadata.json').read_text())
 queue=[];full=list(range(16));pools=[list(range(8)),list(range(8,16))]
 def group(gid,family,workload,anchor,repeat,cells,order,half=0):
  for pos,cell in enumerate(order,1):
   condition,k,schedule,layout=cells[cell];saving=condition.endswith('A')
   train=full if layout=='F' else pools[half]
   writer=(full if layout=='F' else pools[half if layout=='S' else 1-half]) if saving else []
   queue.append(dict(slot_id=f'rep_{gid}_{cell}',group_id=gid,family=family,workload=workload,block=anchor,anchor_block=anchor,repeat_block=repeat,cell_id=cell,condition=condition,interval=k,save_schedule=schedule,layout=layout,half_pool='PQ'[half],training_pool='PQ' if layout=='F' else 'PQ'[half],placement=({'F':'full16','S':'shared8','I':'split8+8'}[layout] if saving else 'absent'),training_cpus=train,writer_cpus=writer,position=pos,schedule_index=len(queue),manifest=f'{AS}/evidence/manifests/{workload}_block_{anchor}.json',anchor=f'{AS}/anchors/{workload}_{anchor}.pt',updates=meta['manifests'][f'{workload}_{anchor}']['measurement_updates'],input_dose=1))
 # Four same-condition repeats per arm, alternating order; excluded from primary contrasts.
 for i,c in enumerate(['OA','O0','O0','OA','OA','O0','O0','OA'],1):
  group(f'D{i:02d}','diagnostic','A',1,i,{c:(c,68 if c=='OA' else 0,'overlap' if c=='OA' else 'none','F')},[c])
 orders=[['O0','P0','OA','PA'],['P0','PA','O0','OA'],['OA','O0','PA','P0'],['PA','OA','P0','O0']]
 for repeat in (1,2):
  for anchor in (1,2,3):
   for work in ('A','U'):
    k=68 if work=='A' else 47
    cells={c:(c,k if c.endswith('A') else 0,'overlap' if c.endswith('A') else 'none','F') for c in orders[0]}
    idx=(repeat-1)*3+anchor-1
    group(f'M{work}{anchor}R{repeat}','initial',work,anchor,repeat,cells,orders[idx%4])
 base_orders=[[0,1,2,3],[1,3,0,2],[2,0,3,1],[3,2,1,0]]
 cells={'OS':('OA',47,'overlap','S'),'PS':('PA',47,'overlap','S'),'OI':('OA',47,'overlap','I'),'PI':('PA',47,'overlap','I')}
 for i in range(6):group(f'P{i+1:02d}','placement','U',1,i+1,cells,[list(cells)[j] for j in base_orders[i%4]],i%2)
 cells={'O47':('OA',47,'overlap','F'),'P47':('PA',47,'overlap','F'),'O94':('OA',94,'overlap','F'),'P94':('PA',94,'overlap','F')}
 for i in range(4):group(f'F{i+1:02d}','frequency','U',1,i+1,cells,[list(cells)[j] for j in base_orders[i]])
 cells={'OASYNC':('OA',47,'overlap','F'),'PASYNC':('PA',47,'overlap','F'),'OWAIT':('OA',47,'serialize','F'),'PWAIT':('PA',47,'serialize','F')}
 for i in range(4):group(f'O{i+1:02d}','overlap','U',1,i+1,cells,[list(cells)[j] for j in base_orders[i]])
 assert len(queue)==112 and len({r['slot_id'] for r in queue})==112
 code={p.name:digest(p) for p in sorted((R/'src').glob('*.py'))}
 old=json.loads((R/'evidence/old_allocation_protocol.json').read_text(encoding='utf-8-sig'))
 for f in ('checkpoint.py','train_core.py','timing.py'):assert code[f]==old['source_sha256'][f],f
 protocol=dict(round_id='s09_replication_20261027',version='controlled_replication_v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),implementation_revision='allocation_core_plus_new_round_contract_v1',primary_clock='CLOCK_MONOTONIC_RAW',budget_clock='host QPC with fixed UTC deadline',source_sha256=code,anchors=meta['anchors'],manifests=meta['manifests'],fixed_queue=queue,training=old['training'],clock_control=json.loads((R/'evidence/gpu_clock_control.json').read_text(encoding='utf-8-sig')),storage='C NVMe-backed WSL ext4; sync_files=True; final archive D USB HDD after measurement',diagnostics='Four A1 OA and four A1 O0 runs; descriptive, never a speed/direction inclusion gate',primary='UltraChat saving-by-input interaction G=(OA-O0)-(PA-P0), A reported separately; average two temporal repetitions within each of three original anchors before anchor-level uncertainty',secondary='U1 placement, frequency and overlap; temporal groups on the same input are not independent dataset samples',failure_policy='First valid result locked. At most 3 actual attempts per slot; retry only documented technical failure. Preserve slow, reversed and high-variance valid runs.',prelaunch='Three 5-second GPU samples; start after idle GPU temp <=50C and range<=2C; ordinary desktop activity permitted. Wait at most 10min per scan cycle then diagnose; never discard a completed run for temperature/speed.',analysis='No old/new pooling. Report full elapsed, fetch, GPU events, checkpoint save/drain, clock/temp/power, host and guest disk I/O clipped to each measured window. Correlation is not causal attribution.',scope='104 main + 8 diagnostic runs. Full-to-split and no-save CPU range are not included in this independent round.',expected=dict(runs=112,main=104,diagnostic=8,final_checkpoints=sum(r['interval']>0 for r in queue),updates=sum(r['updates'] for r in queue),tokens=sum(meta['manifests'][f"{r['workload']}_{r['block']}"]['effective_supervised_tokens'] for r in queue)))
 path=R/'protocol.json'
 if path.exists():raise RuntimeError('Protocol already frozen; do not overwrite')
 path.write_text(json.dumps(protocol,indent=2)+'\n')
 (R/'evidence/protocol_sha256.txt').write_text(digest(path)+'\n')
 print(json.dumps(protocol['expected']))
 return protocol
if __name__=='__main__':build()
