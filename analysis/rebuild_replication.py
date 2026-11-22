"""Rebuild paper numbers and vector figures from retained run tables.

Usage: python analysis/rebuild_replication.py
Dependencies: scipy, matplotlib. No GPU or original experiment directory needed.
"""
from pathlib import Path
import csv, json, math, statistics as st
from collections import defaultdict
from scipy.stats import t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
def readcsv(name):
    return list(csv.DictReader((ROOT / 'data' / name).open(encoding='utf-8-sig')))
def stats(v):
    v = list(v); n = len(v); m = st.mean(v); sd = st.stdev(v)
    h = t.ppf(.975, n-1) * sd / math.sqrt(n)
    return dict(n=n, mean=m, sd=sd, ci=[m-h,m+h], values=v)
def ci(s, digits=2):
    return f"${s['mean']:+.{digits}f}\\; [{s['ci'][0]:.{digits}f}, {s['ci'][1]:.{digits}f}]$"
def write_table(name, lines):
    (ROOT/'tables'/name).write_text('\n'.join(x + r' \\' for x in lines)+'\n',encoding='utf-8')
def contrast(c, family, metric='elapsed_s'):
    z = {k:float(v[metric]) for k,v in c.items()}
    if family == 'initial': a,b = z['OA']-z['O0'],z['PA']-z['P0']
    elif family == 'placement': a,b = z['OS']-z['OI'],z['PS']-z['PI']
    elif family == 'frequency': a,b = z['O47']-z['O94'],z['P47']-z['P94']
    elif family == 'overlap': a,b = z['OASYNC']-z['OWAIT'],z['PASYNC']-z['PWAIT']
    elif family == 'A': a,b = z['OF']-z['OI'],z['PF']-z['PI']
    elif family == 'B': a,b = z['OS']-z['OI'],z['PS']-z['PI']
    elif family == 'C': a,b = z['O0F']-z['O0H'],z['P0F']-z['P0H']
    return dict(online=a,cached=b,interaction=a-b)

new=readcsv('replication_runs.csv'); old=readcsv('historical_main_runs.csv')
assert len(new)==112 and len(old)==156
gg=defaultdict(list)
for r in new:
    if r['family']!='diagnostic': gg[r['group_id']].append(r)
effects={}; initial=defaultdict(list)
for gid, rows in gg.items():
    assert len(rows)==4
    r=rows[0]; c={x['cell_id']:x for x in rows}; f=r['family']
    e=dict(group=gid,family=f,workload=r['workload'],block=r['block'],**contrast(c,f))
    effects[gid]=e
    if f=='initial': initial[(r['workload'],r['block'])].append(e)
newstats={}; pathstats={}; anchors=[]
for w in ['A','U']:
    values=[]
    for b in ['1','2','3']:
        es=initial[(w,b)]; assert len(es)==2
        mean={k:st.mean(e[k] for e in es) for k in ['online','cached','interaction']}
        anchors.append(dict(workload=w,block=b,**mean)); values.append(mean['interaction'])
    newstats['initial_'+w]=stats(values)
for f in ['placement','frequency','overlap']:
    es=[e for e in effects.values() if e['family']==f]
    newstats[f]=stats(e['interaction'] for e in es)
    pathstats[f]={k:stats(e[k] for e in es) for k in ['online','cached']}
review=json.loads((ROOT/'evidence/replication_review.json').read_text(encoding='utf-8-sig'))
for k,v in newstats.items():
    assert math.isclose(v['mean'],review['new_contrasts'][k]['mean'],abs_tol=1e-8)
    assert all(math.isclose(a,b,abs_tol=1e-7) for a,b in zip(v['ci'],review['new_contrasts'][k]['ci95']))

historical={}; histgroups={}; rejected=[]
specs=[('initial_A','finish_v3','initial','Alpaca'),('initial_U','finish_v3','initial','UltraChat'),
       ('frequency','frequency','frequency',None),('overlap','overlap','overlap',None),
       ('placement_R1','placement','placement',None),('full_split','allocation','A',None),
       ('placement_R2','allocation','B',None),('range','allocation','C',None)]
for label,stage,fam,work in specs:
    rr=[x for x in old if x['stage']==stage and (not work or x['workload']==work) and (stage!='allocation' or x['family']==fam)]
    groups=defaultdict(list)
    for x in rr:
        if fam=='placement' and x['condition'] not in ['OS','OI','PS','PI']:continue
        groups[x['asset_block'] if fam=='initial' else x['group_id'] or x['repeat_block']].append(x)
    es=[]
    for gid,rows in groups.items():
        identities={(x['measurement_segment'],x['boot_id'],x['session_id'],x['comparable_implementation']) for x in rows}
        c={x['condition'].replace('_',''):x for x in rows}
        e=dict(group=gid,**contrast(c,fam),fetch_interaction=contrast(c,fam,'fetch_s')['interaction'])
        if len(identities)!=1 or any(x['crosses_user_pause'].lower()=='true' for x in rows):
            rejected.append(dict(series=label,reason='cross-restart comparison',**e));continue
        assert len(rows)==4
        es.append(e)
    histgroups[label]=es
    historical[label]={k:stats(e[k] for e in es) for k in ['online','cached','interaction','fetch_interaction']}
assert len(rejected)==1 and historical['placement_R1']['interaction']['n']==5

means={}; initial_lines=[]; details=[]
for w,wl in [('A','Alpaca'),('U','UltraChat')]:
    means[w]={}
    for c in ['O0','P0','OA','PA']:
        rs=[r for r in new if r['family']=='initial' and r['workload']==w and r['cell_id']==c]
        means[w][c]={m:st.mean(float(r[m]) for r in rs) for m in ['elapsed_s','fetch_s']}
    for a,b,label in [('O0','P0','No save'),('OA','PA','Async')]:
        o,p=means[w][a],means[w][b]
        row=dict(workload=wl,condition=label,online_s=o['elapsed_s'],cached_s=p['elapsed_s'],
                 direct_s=o['elapsed_s']-p['elapsed_s'],direct_pct=100*(1-p['elapsed_s']/o['elapsed_s']),
                 online_fetch_s=o['fetch_s'],cached_fetch_s=p['fetch_s'],fetch_reduction_pct=100*(1-p['fetch_s']/o['fetch_s']))
        details.append(row)
        initial_lines.append(f"{wl} & {label} & {o['elapsed_s']:.2f} & {p['elapsed_s']:.2f} & {row['direct_s']:+.2f} & {o['fetch_s']:.2f} & {p['fetch_s']:.2f}")
write_table('replication_initial.tex', initial_lines)
labels=[('initial_A','Saving: Alpaca','initial_A'),('initial_U','Saving: UltraChat','initial_U'),
        ('placement','Placement R1','placement_R1'),('placement','Placement R2','placement_R2'),
        ('frequency','Frequency','frequency'),('overlap','Overlap','overlap')]
write_table('replication_interactions.tex',[
    f"{label} & {historical[h]['interaction']['n']} & {ci(historical[h]['interaction'])} & {newstats[k]['n']} & {ci(newstats[k])}"
    for k,label,h in labels])
write_table('replication_async.tex',[
    f"{label} & {ci(stats(-x for x in pathstats['overlap'][k]['values']))} & 4/4"
    for k,label in [('online','Online'),('cached','Cached')]])
write_table('layout.tex',[
    f"{label} & {historical[k]['interaction']['n']} & {ci(historical[k]['online'],1)} & {ci(historical[k]['cached'],1)} & {ci(historical[k]['interaction'],1)}"
    for k,label in [('placement_R1','Placement R1'),('full_split',r'Full$\to$split'),('placement_R2','Placement R2'),('range','Range (no save)')]])
write_table('fetch_short.tex',[
    f"{label} & {historical[k]['interaction']['mean']:+.1f} & {historical[k]['fetch_interaction']['mean']:+.3f} & {historical[k]['interaction']['mean']-historical[k]['fetch_interaction']['mean']:+.1f}"
    for k,label in [('placement_R1','Placement R1'),('full_split',r'Full$\to$split'),('placement_R2','Placement R2'),('range','Range')]])
noise={}
oldnoise=json.loads((ROOT/'data/historical_noise_reference.json').read_text())['rows']
for key,rows in [('Historical OA',oldnoise),('Replication OA',[r for r in new if r['family']=='diagnostic' and r['cell_id']=='OA']),('Replication O0',[r for r in new if r['family']=='diagnostic' and r['cell_id']=='O0'])]:
    noise[key]={}
    for m in ['elapsed_s','fetch_s']:
        vals=[float(x[m]) for x in rows];s=stats(vals)
        s.update(range=max(vals)-min(vals),cv=100*s['sd']/s['mean']);noise[key][m]=s
write_table('replication_noise.tex',[
    f"{k} & {v['elapsed_s']['range']:.2f} & {v['elapsed_s']['sd']:.2f} & {v['elapsed_s']['cv']:.2f} & {v['fetch_s']['sd']:.3f}"
    for k,v in noise.items()])
u_repeats={b:[e['interaction'] for e in initial[('U',b)]] for b in ['1','2','3']}
within_ms=sum(sum((v-st.mean(vs))**2 for v in vs) for vs in u_repeats.values())/3
se=math.sqrt(within_ms/6);mu=newstats['initial_U']['mean'];half=t.ppf(.975,3)*se
sensitivity=dict(model='Conditional on three fixed anchors; independent homoscedastic temporal errors',
    repeats=u_repeats,within_mean_square_s2=within_ms,residual_df=3,standard_error_s=se,
    mean_s=mu,ci95_s=[mu-half,mu+half],replaces_primary_analysis=False)
assert math.isclose(within_ms,226.530860412566,abs_tol=1e-7)
drains={}
for cell in ['OASYNC','OWAIT','PASYNC','PWAIT']:
    rows=[r for r in new if r['family']=='overlap' and r['cell_id']==cell];assert len(rows)==4
    drains[cell]={m:st.mean(float(r[m]) for r in rows) for m in ['elapsed_s','post_save_drain_s','final_drain_s']}
out=dict(new_interactions=newstats,new_path_differences=pathstats,new_initial_means=means,
         new_initial_direct=details,new_anchor_effects=anchors,new_group_effects=list(effects.values()),
         historical=historical,historical_groups=histgroups,excluded_historical_groups=rejected,
         diagnostics=noise,within_anchor_sensitivity=sensitivity,overlap_drain_means=drains)
(ROOT/'analysis/paper_statistics.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
with (ROOT/'analysis/replication_groups.csv').open('w',newline='',encoding='utf-8') as f:
    wr=csv.DictWriter(f,fieldnames=list(next(iter(effects.values()))));wr.writeheader();wr.writerows(effects.values())

plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],
    'font.size':9,'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':9,'legend.fontsize':8,
    'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(figsize=(7.16,3.2))
fig.subplots_adjust(left=.21,right=.98,bottom=.18,top=.86)
items=[('initial_A','Saving: Alpaca','initial_A'),('initial_U','Saving: UltraChat','initial_U'),
       ('placement','CPU placement','placement_R1'),('frequency','Frequency','frequency'),('overlap','Overlap','overlap')]
for y,(k,label,h) in enumerate(items):
    for s,dy,col,mark in [(historical[h]['interaction'],-.17,'#666666','o'),(newstats[k],.17,'#0072B2','s')]:
        ax.errorbar(s['mean'],y+dy,xerr=[[s['mean']-s['ci'][0]],[s['ci'][1]-s['mean']]],fmt=mark,color=col,
                    markersize=4.5,capsize=2.5,linewidth=1.1)
    if k=='placement':
        s=historical['placement_R2']['interaction'];ax.errorbar(s['mean'],y,xerr=[[s['mean']-s['ci'][0]],[s['ci'][1]-s['mean']]],fmt='^',color='#D55E00',markersize=4.5,capsize=2.5,linewidth=1.1)
ax.axvline(0,color='black',linewidth=.7,linestyle='--');ax.set_yticks(range(5),[x[1] for x in items]);ax.invert_yaxis()
ax.set_xlabel('Input × intervention contrast (seconds per training window)')
ax.set_xlim(-65,70);ax.set_xticks([-60,-40,-20,0,20,40,60]);ax.grid(axis='x',color='#dddddd',linewidth=.5)
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([],[],marker='o',color='#666666',label='Historical (placement: R1)'),
                   Line2D([],[],marker='^',color='#D55E00',label='Historical placement R2'),
                   Line2D([],[],marker='s',color='#0072B2',label='Controlled replication')],
          loc='lower center',bbox_to_anchor=(.5,1.02),ncol=3,frameon=False,handlelength=1.2,columnspacing=1.4)
import datetime
fig.savefig(ROOT/'figures/replication_interactions.pdf',metadata={'Creator':'Anonymous artifact reconstruction','CreationDate':datetime.datetime(2000,1,1,tzinfo=datetime.timezone.utc)})
fig.savefig(ROOT/'figures/replication_interactions.png',dpi=180);plt.close(fig)
print(json.dumps({'initial_direct':details,'async':pathstats['overlap'],'historical_R1':historical['placement_R1'],
                  'status':'PASS'},indent=2))
