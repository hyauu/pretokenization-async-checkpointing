"""Post-collection analysis. Never invoked concurrently with timed training."""
import pathlib,sys,json,csv,math,statistics,collections,datetime
R=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'host_deps'))
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import t as student_t
D=R/'delivery';D.mkdir(exist_ok=True)
def read(p):return json.loads(pathlib.Path(p).read_text(encoding='utf-8-sig'))
def writecsv(path,rows):
 if not rows:return
 keys=list(dict.fromkeys(k for r in rows for k in r))
 with path.open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def summarise(values):
 a=np.array(values,dtype=float);n=len(a)
 if not n:return {'n':0}
 out=dict(n=n,mean=float(a.mean()),median=float(np.median(a)),minimum=float(a.min()),maximum=float(a.max()),range=float(np.ptp(a)))
 if n>1:
  sd=float(a.std(ddof=1));se=sd/math.sqrt(n);half=float(student_t.ppf(.975,n-1)*se)
  out.update(sd=sd,cv_percent=100*sd/float(a.mean()) if a.mean()!=0 else None,ci95=[float(a.mean()-half),float(a.mean()+half)])
 return out
def main():
 state=read(R/'state.json');assert state['status'] in ('COLLECTION_FINISHED','BUDGET_COLLECTION_CLOSED','ANALYZING','ARCHIVING','READY_FOR_VISUAL_REVIEW','DELIVERED')
 p=read(R/'protocol.json');queue={x['slot_id']:x for x in p['fixed_queue']}
 ledger=[json.loads(x) for x in (R/'evidence/gpu_ledger.jsonl').read_text().splitlines()]
 accepted={x['slot_id']:x for x in ledger if x.get('status')=='VALID'}
 rows=[];details={};inventory=[]
 for slot,entry in accepted.items():
  r=queue[slot];out=R/'runs'/entry['attempt_id'];v=read(out/'result.json');validation=read(out/'host_validation.json');assert validation['status']=='PASS'
  records=v['update_records'];inv=read(out/'checkpoint_inventory.json')['checkpoints']
  periodic=[c for c in inv if not c['initial']]
  row={k:r[k] for k in ('slot_id','group_id','family','workload','block','repeat_block','cell_id','condition','layout','training_pool')}
  row.update(attempt_id=entry['attempt_id'],session_id=v['session_id'],boot_id=v['boot_id'],implementation=v['comparable_implementation'],elapsed_s=v['elapsed_s'],fetch_s=sum(x['fetch_s'] for x in records),gpu_event_s=sum(x.get('gpu_event_ms',0) for x in records)/1000,staging_wait_s=sum(x['staging_wait_s'] for x in records),save_call_s=sum(x['save_call_s'] for x in records),post_save_drain_s=sum(x['post_save_drain_s'] for x in records),final_drain_s=v['final_drain_s'],updates=v['completed_updates'],tokens=v['effective_supervised_tokens'],periodic_commits=v['checkpoint_commits'],initial_commits=v['initial_checkpoint_commits'],checkpoint_periodic_logical_bytes=sum(c['bytes'] for c in periodic),checkpoint_all_logical_bytes=sum(c['bytes'] for c in inv),start_qpc=validation['timing']['start_qpc'],end_qpc=validation['timing']['end_qpc'])
  rows.append(row);details[slot]=v
  for c in inv:inventory.append(dict(attempt_id=entry['attempt_id'],**c))
 gpu=[json.loads(line) for line in (R/'evidence/gpu.jsonl').read_text().splitlines()]
 host=[json.loads(line) for line in (R/'evidence/host_resources.jsonl').read_text().splitlines()]
 for r in rows:
  win=[x for x in gpu if r['start_qpc']<=x['qpc']<=r['end_qpc']]
  for field,tag in [('temperature.gpu','temperature_c'),('clocks.sm','sm_clock_mhz'),('power.draw','power_w'),('utilization.gpu','gpu_util_pct')]:
   vals=[]
   for x in win:
    try:vals.append(float(x[field]))
    except (ValueError,KeyError):pass
   if vals:r[tag+'_mean']=statistics.mean(vals);r[tag+'_min']=min(vals);r[tag+'_max']=max(vals)
  r['gpu_samples']=len(win)
  for f in ('sw_power_cap','hw_thermal_slowdown','sw_thermal_slowdown','reliability'):
   r[f+'_active_samples']=sum(x.get('clocks_event_reasons.'+f)=='Active' for x in win)
  hw=[x for x in host if r['start_qpc']<=x['qpc']<=r['end_qpc']]
  if len(hw)>1:
   a,b=hw[0],hw[-1];r['host_io_covered_s']=(b['qpc']-a['qpc'])/b['qpc_frequency']
   for disk,values in b['disk'].items():
    old=a['disk'].get(disk,{})
    if 'write_bytes' in old:r['host_'+disk+'_write_bytes']=values['write_bytes']-old['write_bytes']
  gr=[json.loads(x) for x in (R/'runs'/r['attempt_id']/'guest_resources.jsonl').read_text().splitlines()]
  v=details[r['slot_id']];gr=[x for x in gr if v['measurement_start_raw_ns']<=x['raw_ns']<=v['measurement_end_raw_ns']]
  if len(gr)>1:
   a,b=gr[0],gr[-1];r['guest_io_covered_s']=(b['raw_ns']-a['raw_ns'])/1e9
   for disk,values in b['disk'].items():
    old=a['disk'].get(disk,{})
    if 'write_bytes' in old:r['guest_'+disk+'_write_bytes']=values['write_bytes']-old['write_bytes']
 writecsv(D/'runs.csv',rows)
 effects=[];group_audit=[]
 for gid in dict.fromkeys(x['group_id'] for x in p['fixed_queue']):
  expected=[x for x in p['fixed_queue'] if x['group_id']==gid];rr=[r for r in rows if r['group_id']==gid]
  complete=len(rr)==len(expected);comparable=complete and len({(r['session_id'],r['boot_id'],r['implementation']) for r in rr})==1
  group_audit.append(dict(group_id=gid,family=expected[0]['family'],expected=len(expected),valid=len(rr),group_complete=complete,contrast_eligible=comparable and len(rr)==4))
  if not comparable or len(rr)!=4:continue
  cells={r['cell_id']:r for r in rr};T={k:r['elapsed_s'] for k,r in cells.items()};fam=rr[0]['family'];extra={}
  if fam=='initial':
   online=T['OA']-T['O0'];cached=T['PA']-T['P0'];value=online-cached;name='G';extra=dict(online_saving_cost_s=online,cached_saving_cost_s=cached,online_both_slower=online>0,cached_both_slower=cached>0)
  elif fam=='placement':
   online=T['OS']-T['OI'];cached=T['PS']-T['PI'];value=online-cached;name='placement_interaction';extra=dict(online_shared_minus_split_s=online,cached_shared_minus_split_s=cached)
  elif fam=='frequency':value=(T['O47']-T['P47'])-(T['O94']-T['P94']);name='frequency_interaction';extra=dict(online_47_minus_94_s=T['O47']-T['O94'],cached_47_minus_94_s=T['P47']-T['P94'])
  else:value=(T['OASYNC']-T['PASYNC'])-(T['OWAIT']-T['PWAIT']);name='overlap_interaction';extra=dict(online_async_minus_wait_s=T['OASYNC']-T['OWAIT'],cached_async_minus_wait_s=T['PASYNC']-T['PWAIT'])
  effects.append(dict(group_id=gid,family=fam,workload=rr[0]['workload'],block=rr[0]['block'],repeat_block=rr[0]['repeat_block'],contrast=name,seconds=value,**extra))
 writecsv(D/'group_audit.csv',group_audit);writecsv(D/'contrasts.csv',effects)
 summaries={};anchor_rows=[]
 for work in ('A','U'):
  anchor_means=[]
  for b in (1,2,3):
   es=[e['seconds'] for e in effects if e['family']=='initial' and e['workload']==work and e['block']==b]
   if len(es)==2:
    value=statistics.mean(es);anchor_means.append(value);anchor_rows.append(dict(workload=work,block=b,temporal_repeats=2,mean_G_s=value))
  summaries['initial_'+work]=dict(unit='anchor mean of two temporal repeats',**summarise(anchor_means))
 for fam in ('placement','frequency','overlap'):
  summaries[fam]=dict(unit='temporal matched group, same U1 input',**summarise([e['seconds'] for e in effects if e['family']==fam]))
 writecsv(D/'anchor_means.csv',anchor_rows)
 historical=[]
 for f in sorted((R/'evidence/historical_diagnostics').glob('*.json')):
  v=read(f);historical.append(dict(attempt_id=f.stem,elapsed_s=v['elapsed_s'],fetch_s=sum(x['fetch_s'] for x in v['update_records'])))
 diagnostics={c:summarise([r['elapsed_s'] for r in rows if r['family']=='diagnostic' and r['condition']==c]) for c in ('O0','OA')}
 diagnostics['historical_OA']=summarise([r['elapsed_s'] for r in historical]);writecsv(D/'historical_diagnostic_reference.csv',historical)
 summary=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),valid_runs=len(rows),expected_runs=112,main_runs=sum(r['family']!='diagnostic' for r in rows),diagnostic_runs=sum(r['family']=='diagnostic' for r in rows),failed_attempts=[x for x in ledger if x.get('status')=='FAILED'],missing_slots=[k for k in queue if k not in accepted],updates=sum(r['updates'] for r in rows),tokens=sum(r['tokens'] for r in rows),logical_checkpoint_bytes=sum(c['bytes'] for c in inventory),final_checkpoints=sum(bool(read(R/'runs'/r['attempt_id']/'retention.json')['retained_final']) for r in rows),diagnostics=diagnostics,contrasts=summaries)
 (D/'summary.json').write_text(json.dumps(summary,indent=2))
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
 fig,axes=plt.subplots(1,2,figsize=(11,4.2),layout='constrained')
 groups=[('Historical saved',historical),('New saved',[r for r in rows if r['family']=='diagnostic' and r['condition']=='OA']),('New no-save',[r for r in rows if r['family']=='diagnostic' and r['condition']=='O0'])]
 for i,(label,rr) in enumerate(groups):
  for ax,key in zip(axes,['elapsed_s','fetch_s']):
   y=[r[key] for r in rr];ax.scatter(np.linspace(i-.08,i+.08,len(y)),y,s=40)
   if y:ax.hlines(statistics.mean(y),i-.17,i+.17,color='black',lw=1.3)
 for ax,title in zip(axes,['Complete measured window (s)','Summed fetch time (s)']):ax.set_xticks(range(3),[x[0] for x in groups],rotation=12);ax.set_title(title);ax.grid(axis='y',alpha=.2)
 fig.suptitle('Same-input repeated runs: all observations shown; mean marked')
 fig.savefig(D/'noise_diagnostics.png');fig.savefig(D/'noise_diagnostics.pdf');plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(11,4.4),layout='constrained')
 for ax,names in zip(axes,[['initial_A','initial_U'],['placement','frequency','overlap']]):
  for i,name in enumerate(names):
   s=summaries[name]
   if not s['n']:continue
   mean=s['mean'];ci=s.get('ci95',[mean,mean]);ax.errorbar(mean,i,xerr=[[mean-ci[0]],[ci[1]-mean]],fmt='o',capsize=4)
  labels=[('Alpaca anchor means' if n=='initial_A' else 'UltraChat anchor means' if n=='initial_U' else n)+f" (n={summaries[n]['n']})" for n in names]
  ax.set_yticks(range(len(names)),labels);ax.axvline(0,color='gray',ls='--');ax.set_xlabel('Interaction contrast (s), mean and 95% t interval');ax.grid(axis='x',alpha=.2)
 axes[0].set_title('Saving × input path');axes[1].set_title('U1 follow-ups: temporal groups')
 fig.savefig(D/'main_contrasts.png');fig.savefig(D/'main_contrasts.pdf');plt.close(fig)
 text=['# S09 C 盘受控复现结果','',f"有效完成 {len(rows)}/112 次，其中主对照 {summary['main_runs']} 次、诊断 {summary['diagnostic_runs']} 次；失败尝试 {len(summary['failed_attempts'])} 次。缺失槽位见 summary.json。",'',f"实际有效工作量：{summary['updates']:,} updates，{summary['tokens']:,} supervised tokens。保存逻辑字节总和 {summary['logical_checkpoint_bytes']/1e9:.3f} GB；保留最终 checkpoint {summary['final_checkpoints']} 个。",'', '## 噪声诊断','']
 for name,s in diagnostics.items():text.append(f"- {name}: n={s['n']}; "+(f"均值 {s['mean']:.3f}s，范围 {s['range']:.3f}s，SD {s.get('sd',float('nan')):.3f}s。" if s['n'] else '无有效结果。'))
 text+=['','历史与新轮为不同时期、不同受控环境，差异不能单独归因于某一个设置。锁频操作被权限拒绝，本轮没有锁频。C 盘 SSD、温度前置条件及完整监控提高可解释性，但不能保证噪声已经消除。','', '## 主要对照','']
 for name,s in summaries.items():
  text.append(f"- {name}: n={s['n']}; "+(f"均值 {s['mean']:+.3f}s，95% t 区间 {s.get('ci95')}；单位为 {s['unit']}。" if s['n'] else '无完整可比单元。'))
 text+=['','初始主比较 G=(OA−O0)−(PA−P0)。正 G 仅表示保存代价的路径差异，必须结合两条路径各自的保存代价；正交互不表示两条路径都加速。后续对比的两条路径绝对差均列于 contrasts.csv。','', '每个初始输入块的两个时间重复先平均，独立输入块数最多为 3；U1 后续组是同输入的时间重复。诊断没有混入主要对照；历史 156 条也没有拼入新的区间估计。','', '磁盘 I/O 为窗口内首尾采样差并报告实际覆盖秒数，可能少覆盖边界。保存文件逻辑字节、操作系统 I/O 与 SSD 物理写入不同。GPU event、fetch、保存等部分时间可能重叠，不应相加作为互斥分解。','', '图像与最终研究判断须由后续审阅核验；尚未据本报告自动改写论文。']
 (D/'RESULTS_zh.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
 print(json.dumps({k:summary[k] for k in ['valid_runs','main_runs','diagnostic_runs','final_checkpoints']}))
if __name__=='__main__':main()
