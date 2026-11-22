"""Offline analysis of retained observations; never imports training/CUDA libraries.

All contrasts are seconds per prescribed training window. No run is replaced.
Student intervals and noncentral-t sensitivity assume independent normal blocks.
"""
import argparse, csv, json, math, platform, statistics as st, os, tempfile
from collections import defaultdict
from pathlib import Path
import scipy
from scipy.stats import t, nct, binomtest
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[1]
def read(p):
    with p.open(encoding='utf-8-sig', newline='') as f: return list(csv.DictReader(f))
def write(p, rows):
    with p.open('w', encoding='utf-8', newline='') as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
def stats(v):
    n=len(v); mean=st.mean(v); sd=st.stdev(v); se=sd/math.sqrt(n); h=t.ppf(.975,n-1)*se
    return dict(n=n,mean_s=mean,sd_s=sd,se_s=se,ci_low_s=mean-h,ci_high_s=mean+h,
                positive=sum(x>0 for x in v),negative=sum(x<0 for x in v),
                p_t=float(2*t.sf(abs(mean/se),n-1)))
def close(a,b,tol=1e-6):
    assert abs(a-b)<tol,(a,b)
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--skip-figure',action='store_true'); args=ap.parse_args()
    out=ROOT/'data/icict'; out.mkdir(exist_ok=True,parents=True)
    rows=read(ROOT/'data/main_runs.csv'); old=read(ROOT/'data/effects.csv')
    assert len(rows)==156 and len(old)==36
    groups=defaultdict(dict)
    for row in rows:
        block=row['group_id'] if row['stage']=='allocation' else row['repeat_block']
        key=(row['stage'],row['design'],row['workload'],row['family'],block)
        assert row['condition'] not in groups[key]; groups[key][row['condition']]=row
    maps={'finish_v3':('OA','PA','O0','P0'), 'frequency':('O47','P47','O94','P94'),
          'overlap':('O_ASYNC','P_ASYNC','O_WAIT','P_WAIT'),
          'placement':('OS','PS','OI','PI'), 'A':('OF','PF','OI','PI'),
          'B':('OS','PS','OI','PI'), 'C':('O0F','P0F','O0H','P0H')}
    effects=[]; fetch=[]; asyncrows=[]; initial=[]; checks=0
    for key,g in sorted(groups.items()):
        stage,design,workload,family,block=key; codes=maps[family if stage=='allocation' else stage]
        a,b,c,d=[g[z] for z in codes]
        val=lambda row,k:float(row[k])
        delta=lambda k:val(a,k)-val(b,k)-val(c,k)+val(d,k)
        src=';'.join(z['source_csv']+':'+z['source_row'] for z in [a,b,c,d])
        series=(family if stage=='allocation' else stage)
        if stage=='finish_v3':series+='_'+workload
        r=dict(stage=stage,design=design,workload=workload,family=family,group_id=block,series=series,
               D_reference_s=val(a,'elapsed_s')-val(b,'elapsed_s'),
               D_target_s=val(c,'elapsed_s')-val(d,'elapsed_s'),interaction_s=delta('elapsed_s'),
               condition_order=';'.join(codes),source_rows=src)
        matches=[z for z in old if z['stage']==stage and z['design']==design and z['workload']==workload and z['block']==block]
        assert len(matches)==1
        close(r['interaction_s'],float(matches[0]['interaction_s'])); checks+=1;effects.append(r)
        if stage in ['allocation','placement']:
            if stage=='allocation':
                assert len({z['session_id'] for z in g.values()})==len({z['boot_id'] for z in g.values()})==1
            f={k:r[k] for k in ['design','family','group_id','series']}
            for field,label in [('elapsed_s','total'),('fetch_s','fetch')]:
                f['R_O_'+label+'_s']=val(a,field)-val(c,field)
                f['R_P_'+label+'_s']=val(b,field)-val(d,field)
                f['interaction_'+label+'_s']=delta(field)
                close(f['R_O_'+label+'_s']-f['R_P_'+label+'_s'],delta(field)); checks+=1
            f['interaction_remainder_s']=f['interaction_total_s']-f['interaction_fetch_s']
            f.update(session_id=';'.join(sorted({z['session_id'] for z in g.values()})),
                     boot_id=';'.join(sorted({z['boot_id'] for z in g.values()})),
                     measurement_segments=';'.join(sorted({z['measurement_segment'] for z in g.values()})),
                     crosses_pause=stage=='placement' and block=='3',source_rows=src)
            fetch.append(f)
        if stage=='overlap':
            asyncrows.append(dict(block=block,O_s=val(c,'elapsed_s')-val(a,'elapsed_s'),P_s=val(d,'elapsed_s')-val(b,'elapsed_s'),source_rows=src))
        if stage=='finish_v3':
            initial.append(dict(workload=workload,block=block,**{z:val(g[z],'elapsed_s') for z in ['O0','P0','OA','PA']},
                D_nosave_s=r['D_target_s'],D_async_s=r['D_reference_s'],interaction_s=r['interaction_s'],
                fetch_online_s=val(a,'fetch_s'),fetch_cached_s=val(b,'fetch_s')))
    summaries=[]
    for series in ['frequency','overlap','placement','A','B','C']:
        v=[r['interaction_s'] for r in effects if r['series']==series]; s=stats(v)
        power=lambda delta:nct.sf(t.ppf(.975,s['n']-1),s['n']-1,delta*math.sqrt(s['n'])/s['sd_s'])+nct.cdf(-t.ppf(.975,s['n']-1),s['n']-1,delta*math.sqrt(s['n'])/s['sd_s'])
        # ncp <= 6 brackets the root for n=4/6 without evaluating extreme tails.
        mde=brentq(lambda d:power(d)-.8,0,6*s['sd_s']/math.sqrt(s['n']))
        summaries.append(dict(series=series,**s,mde80_s=mde,mde_pct_of_395s=100*mde/395))
    layout=[]
    for series in ['placement','A','B','C']:
        ff=[r for r in fetch if r['series']==series]
        for metric in ['R_O_total_s','R_P_total_s','interaction_total_s','R_O_fetch_s','R_P_fetch_s','interaction_fetch_s','interaction_remainder_s']:
            layout.append(dict(series=series,metric=metric,**stats([r[metric] for r in ff])))
    pooled=[]
    for exclude in [False,True]:
        ff=[r for r in fetch if r['series'] in ['placement','B'] and not (exclude and r['crosses_pause'])]
        v=[r['interaction_total_s'] for r in ff]; s=stats(v)
        pooled.append(dict(selection='excluding_old_block_3' if exclude else 'all_two_rounds',**s,
            p_sign=binomtest(s['positive'],s['n'],.5).pvalue,posthoc=True))
    asyncsummary=[dict(path=p,**stats([r[p+'_s'] for r in asyncrows])) for p in ['O','P']]
    for series,expected in [('placement',17.6401352969997),('A',6.523510487997384),('B',15.97757634083311),('C',-19.37536403775027)]:
        # Old summary in the plan is rounded; check its six-decimal reference below.
        got=next(r['mean_s'] for r in summaries if r['series']==series)
        close(got,expected,1e-6)
    expectedfetch={'placement':.334809,'A':-.179347,'B':.096308,'C':.054469}
    for s,e in expectedfetch.items():close(next(r['mean_s'] for r in layout if r['series']==s and r['metric']=='interaction_fetch_s'),e,1e-6)
    for name,data in [('effects',effects),('fetch_decomposition',fetch),('layout_summary',layout),('sensitivity',summaries),('posthoc_pooling',pooled),('async_wait_blocks',asyncrows),('async_wait_summary',asyncsummary),('initial_results',initial)]:write(out/(name+'.csv'),data)
    totals={k:sum(int(r[k]) for r in rows) for k in ['updates','supervised_tokens','periodic_commits','initial_commits']}
    assert totals==dict(updates=45504,supervised_tokens=587858856,periodic_commits=672,initial_commits=116)
    ancillary={}
    for w in ['Alpaca','UltraChat']:
        z=[r for r in initial if r['workload']==w]; o=st.mean(r['fetch_online_s'] for r in z);p=st.mean(r['fetch_cached_s'] for r in z)
        ancillary[w]=dict(fetch_online_s=o,fetch_cached_s=p,fetch_reduction_pct=100*(o-p)/o)
    es=[]
    for key,g in groups.items():
        if key[0]=='placement':es.append((float(g['OS']['elapsed_s'])-float(g['PS']['elapsed_s']))-(float(g['O0']['elapsed_s'])-float(g['P0']['elapsed_s'])))
    ancillary['old_shared_save_increment']=dict(mean_s=st.mean(es),median_s=st.median(es),positive=sum(x>0 for x in es),values_s=es)
    ancillary['frequency_loo_excluding_block4_s']=st.mean(r['interaction_s'] for r in effects if r['series']=='frequency' and r['group_id']!='4')
    report=dict(status='PASS',main_runs=156,complete_effects=36,layout_groups=22,arithmetic_checks=checks,totals=totals,
                python=platform.python_version(),scipy=scipy.__version__,ancillary=ancillary,
                scope='Offline reconstruction; no training. Statistical intervals are model-based descriptive summaries, unadjusted for multiplicity.')
    (out/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    # Tables consumed by the manuscript; all numbers are generated here.
    tex=ROOT/'tables';tex.mkdir(exist_ok=True)
    def savet(name,lines):(tex/name).write_text('\n'.join(lines)+'\n')
    savet('initial.tex',[f"{r['workload'][0]}{r['block']} & {r['O0']:.1f} & {r['P0']:.1f} & {r['OA']:.1f} & {r['PA']:.1f} & {r['D_async_s']:+.1f} & {r['interaction_s']:+.1f} \\\\" for r in initial])
    labels={'placement':'Placement R1','A':'Full$\\to$split','B':'Placement R2','C':'Range (no save)'}
    lines=[]
    for s in labels:
        rr=[next(r for r in layout if r['series']==s and r['metric']==m) for m in ['R_O_total_s','R_P_total_s','interaction_total_s']]
        cell=lambda r:f"${r['mean_s']:+.1f}\\; [{r['ci_low_s']:.1f}, {r['ci_high_s']:.1f}]$"
        lines.append(labels[s]+' & '+str(rr[0]['n'])+' & '+' & '.join(cell(r) for r in rr)+' \\\\')
    savet('layout.tex',lines)
    savet('fetch.tex',[labels[s]+' & '+' & '.join(f"{next(r['mean_s'] for r in layout if r['series']==s and r['metric']==m):+.3f}" for m in ['interaction_total_s','interaction_fetch_s','interaction_remainder_s'])+' \\\\' for s in labels])
    shortlabels={'placement':'Placement R1','A':'Full$\\to$split','B':'Placement R2','C':'Range'}
    savet('fetch_short.tex',[shortlabels[s]+' & '+' & '.join(format(next(r['mean_s'] for r in layout if r['series']==s and r['metric']==m),'+.3f' if m=='interaction_fetch_s' else '+.1f') for m in ['interaction_total_s','interaction_fetch_s','interaction_remainder_s'])+' \\\\' for s in labels])
    savet('async.tex',[str(r['block'])+f" & {r['O_s']:+.1f} & {r['P_s']:+.1f} \\\\" for r in asyncrows])
    savet('sensitivity.tex',[r['series']+f" & {r['n']} & {r['mean_s']:+.3f} & [{r['ci_low_s']:.3f}, {r['ci_high_s']:.3f}] & {r['mde80_s']:.3f} \\\\" for r in summaries])
    if not args.skip_figure:
        # Draw from the exported tables, so the figure and distributed data agree.
        fetch=read(out/'fetch_decomposition.csv')
        for r in fetch:
            for k in r:
                if k.endswith('_s'):r[k]=float(r[k])
            r['crosses_pause']=r['crosses_pause']=='True'
        summaries=read(out/'sensitivity.csv')
        for r in summaries:
            for k in ['mean_s','ci_low_s','ci_high_s']:r[k]=float(r[k])
        os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'checkpoint-paper-mpl'))
        import matplotlib;matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'svg.fonttype':'none'})
        figs=ROOT/'figures';figs.mkdir(exist_ok=True)
        fig,axes=plt.subplots(1,2,figsize=(7.1,3.1),gridspec_kw={'width_ratios':[1.08,1]})
        colors=['#365f91','#b25b18','#248378','#82599a']
        for i,(s,color) in enumerate(zip(labels,colors)):
            ff=[r for r in fetch if r['series']==s]; vals=[r['interaction_total_s'] for r in ff]
            offsets=[(j-(len(ff)-1)/2)*.065 for j in range(len(ff))]
            for f,off in zip(ff,offsets):axes[0].plot(f['interaction_total_s'],i+off,'D' if f['crosses_pause'] else 'o',ms=5,color=color,markerfacecolor='white' if f['crosses_pause'] else color)
            sr=next(r for r in summaries if r['series']==s)
            axes[0].errorbar(sr['mean_s'],i+.27,xerr=[[sr['mean_s']-sr['ci_low_s']],[sr['ci_high_s']-sr['mean_s']]],fmt='s',ms=4,capsize=3,color='black')
            total=st.mean(vals);fe=st.mean(r['interaction_fetch_s'] for r in ff)
            axes[1].barh(i-.13,total,height=.24,color=color)
            axes[1].barh(i+.13,fe,height=.24,color='white',edgecolor=color,hatch='////')
            axes[1].text(fe+(1 if fe>=0 else -1),i+.13,f'{fe:+.3f}',ha='left' if fe>=0 else 'right',va='center',fontsize=8)
        for ax in axes:
            ax.axvline(0,color='#555555',lw=.8);ax.set_yticks(range(4),['R1','Full\u2192split','R2','Range']);ax.invert_yaxis();ax.spines[['top','right']].set_visible(False);ax.grid(axis='x',alpha=.15)
        axes[0].set_xlabel('Group interaction (s)');axes[0].set_title('(a) Blocks and mean t95 intervals',fontsize=10)
        axes[1].set_xlabel('Mean interaction (s)');axes[1].set_title('(b) Total (solid) and fetch (hatched)',fontsize=10)
        axes[1].set_xlim(-24,24);fig.tight_layout(pad=.7)
        fig.savefig(figs/'layout_fetch.pdf',metadata={'CreationDate':None,'Creator':'Offline analysis'});fig.savefig(figs/'layout_fetch.png',dpi=220);plt.close(fig)
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
