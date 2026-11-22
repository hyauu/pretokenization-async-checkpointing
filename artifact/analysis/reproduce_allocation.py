#!/usr/bin/env python3
"""Reconstruct the allocation A/B/C results from packaged original fields.

No training, GPU, network, external review directory, or experiment code is used.
The three comparison families remain separate throughout the analysis.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import statistics
import tempfile

from reproduce_prior import MAIN_COLUMNS, EFFECT_COLUMNS

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'
ORIGINAL = DATA / 'original' / 'allocation'
TOKENS = 3975934
# Two-sided 95% Student-t quantiles. The intervals are explicitly approximate.
T975 = {2: 12.706204736432095, 3: 4.302652729696142,
        4: 3.182446305284263, 6: 2.570581835636314}
CELLS = {'A': ['OF', 'OI', 'PF', 'PI'], 'B': ['OS', 'OI', 'PS', 'PI'],
         'C': ['O0F', 'O0H', 'P0F', 'P0H']}
GROUP_COUNTS = {'A': 6, 'B': 6, 'C': 4}
INTERACTIONS = {'A': 'K_A', 'B': 'J_B', 'C': 'K_C'}
SETTINGS = {'A': ('full16', 'split8+8'), 'B': ('shared8', 'split8+8'),
            'C': ('no_save16', 'no_save8')}
RUN_EXTRA = ('attempt_id placement training_pool measurement_segment boot_id '
             'crosses_user_pause training_cpus writer_cpus source_result '
             'original_result_path original_result_sha256 pre_first_save_s final_drain_s '
             'family group_id layout half_pool session_id comparable_implementation '
             'schedule_index raw_start_ns raw_end_ns').split()
EFFECT_EXTRA = ('family group_id half_pool session_id boot_id measurement_segment '
                'comparable_implementation group_complete contrast_eligible '
                'crosses_user_pause direct_reference_setting direct_target_setting '
                'R_O R_P K_A J_B K_C R_O_pct_of_reference R_P_pct_of_reference '
                'OF OI PF PI OS PS O0F O0H P0F P0H').split()


class Audit:
    def __init__(self):
        self.identities = 0
        self.comparisons = []

    def equal(self, actual, expected, label):
        self.identities += 1
        if actual != expected:
            raise ValueError(f'{label}: {actual!r} != {expected!r}')

    def close(self, actual, expected, label):
        if isinstance(actual, list):
            self.equal(len(actual), len(expected), label + ': length')
            for i, (a, b) in enumerate(zip(actual, expected)):
                self.close(a, b, f'{label}[{i}]')
            return
        error = abs(float(actual) - float(expected))
        if not math.isfinite(error) or error > 1e-7:
            raise ValueError(f'{label}: {actual!r} != {expected!r}, error={error}')
        self.comparisons.append({'label': label, 'absolute_error': error})


def read_csv(name):
    with (ORIGINAL / name).open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        rows, previous_end = [], 1
        for row in reader:
            row['_source_row'] = previous_end + 1
            previous_end = reader.line_num
            rows.append(row)
        return rows


def read_json(name):
    return json.loads((ORIGINAL / name).read_text(encoding='utf-8-sig'))


def write_csv(path, columns, rows):
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns, lineterminator='\n')
        writer.writeheader()
        writer.writerows({key: row.get(key, '') for key in columns} for row in rows)


def descriptive(values):
    values = list(values)
    n = len(values)
    mean = statistics.fmean(values)
    sd = statistics.stdev(values)
    half = T975[n] * sd / math.sqrt(n)
    return {'n': n, 'values': values, 'mean': mean, 'median': statistics.median(values),
            'min': min(values), 'max': max(values), 'sd': sd,
            'positive': sum(v > 0 for v in values), 'negative': sum(v < 0 for v in values),
            'zero': sum(v == 0 for v in values), 't95': [mean-half, mean+half],
            'leave_one_means': [statistics.fmean(values[:i]+values[i+1:]) for i in range(n)]}


def scope(family):
    return (f'{GROUP_COUNTS[family]} temporal four-condition groups of fixed U1; '
            'each analyzed group uses one session and boot; family ' + family)


def verify_sources(audit):
    manifest = json.loads((DATA / 'allocation_source_manifest.json').read_text())
    audit.equal(len(manifest), 75, 'frozen allocation inputs')
    for item in manifest:
        path = ROOT / item['package_path']
        audit.equal(hashlib.sha256(path.read_bytes()).hexdigest(), item['sha256'],
                    item['package_path'] + ': packaged SHA256')
    return manifest


def reconstruct_runs(audit):
    order = read_csv('order.csv')
    planned = {r['slot_id']: r for r in order}
    refs = {r['attempt_id']: r for r in read_csv('runs.csv')}
    slots = {r['slot_id']: r for r in read_csv('slots.csv')}
    state = {r['slot_id']: r for r in read_json('state.json')['slots']}
    attempts = read_csv('attempts.csv')
    attempts_by_id = {r['attempt_id']: r for r in attempts}
    audit.equal(len(planned), 64, 'unique planned slots')
    audit.equal(len(refs), 64, 'selected runs')
    audit.equal(len(attempts), 65, 'attempt records')
    audit.equal(len(attempts_by_id), 65, 'unique attempts')
    outputs, groups, starts, scientific_protocols = [], defaultdict(dict), set(), set()
    files = sorted((ORIGINAL / 'results').glob('*.core.json'))
    audit.equal(len(files), 64, 'core result records')
    for path in files:
        r = json.loads(path.read_text())
        attempt, slot = r['attempt_id'], r['slot_id']
        row, plan = refs[attempt], planned[slot]
        request, process_group = r['_request_fields'], r['_process_group_fields']
        for flag in ['valid', 'training_valid', 'timing_valid', 'placement_valid']:
            audit.equal(r[flag], True, attempt + ': ' + flag)
        audit.equal(r['primary_clock'], 'CLOCK_MONOTONIC_RAW', attempt + ': clock')
        for key in ['slot_id', 'group_id', 'family', 'cell_id', 'half_pool', 'layout',
                    'condition', 'save_schedule', 'training_pool']:
            audit.equal(str(r[key]), plan[key], attempt + ': frozen ' + key)
            audit.equal(r[key], request[key], attempt + ': request ' + key)
        for key in ['schedule_index', 'position', 'input_dose', 'block', 'repeat_block', 'interval']:
            audit.equal(r[key], int(plan[key]), attempt + ': frozen ' + key)
            audit.equal(r[key], request[key], attempt + ': request ' + key)
        for key in ['training_cpus', 'writer_cpus']:
            audit.equal(r[key], json.loads(plan[key]), attempt + ': frozen ' + key)
            audit.equal(r[key], request[key], attempt + ': request ' + key)
        for key in ['session_id', 'measurement_segment', 'comparable_implementation', 'execution_protocol_sha256']:
            audit.equal(r[key], request[key], attempt + ': request ' + key)
            audit.equal(str(r[key]), row[key], attempt + ': CSV ' + key)
        audit.equal(r['boot_id'], process_group['boot_id'], attempt + ': process boot')
        audit.equal(r['boot_id'], row['boot_id'], attempt + ': CSV boot')
        first_valid = [a['attempt_id'] for a in state[slot]['attempts'] if a.get('valid')][0]
        audit.equal(attempt, first_valid, slot + ': first valid attempt')
        full_original_sha = r['_provenance']['original_source_sha256']
        audit.equal(row['result_sha256'], full_original_sha, attempt + ': original result identity')
        for selection in [slots[slot], state[slot]]:
            audit.equal(selection['selected_attempt'], attempt, slot + ': selected attempt')
            audit.equal(selection['selected_result_sha256'], full_original_sha, slot + ': selected identity')
        audit.equal(attempts_by_id[attempt]['adopted'], 'True', attempt + ': adopted')
        records = r['update_records']
        audit.equal(len(records), 282, attempt + ': update records')
        audit.equal(r['completed_updates'], 282, attempt + ': updates')
        audit.equal(r['effective_supervised_tokens'], TOKENS, attempt + ': tokens')
        audit.equal(sum(u['tokens'] for u in records), TOKENS, attempt + ': summed tokens')
        audit.equal(r['block'], 1, attempt + ': U1')
        audit.equal(r['final_global_step'], 285, attempt + ': final step')
        audit.equal(r['final_consumed_cursor'], 2280, attempt + ': final cursor')
        starts.add(r['measurement_initial_state']['complete_digest'])
        scientific_protocols.add(r['protocol_sha256'])
        saving = r['family'] != 'C'
        audit.equal(r['checkpoint_commits'], 6 if saving else 0, attempt + ': periodic saves')
        audit.equal(r['initial_checkpoint_commits'], 1 if saving else 0, attempt + ': initial saves')
        elapsed = (r['measurement_end_raw_ns'] - r['measurement_start_raw_ns']) / 1e9
        fetch = sum(u['fetch_s'] for u in records)
        prefix = records[46]['end'] - r['measurement_start_raw_s']
        audit.close(elapsed, r['elapsed_s'], attempt + ': integer RAW versus result')
        audit.close(elapsed, row['T_raw_s'], attempt + ': integer RAW versus CSV')
        audit.close(fetch, row['fetch_s'], attempt + ': fetch sum')
        audit.close(prefix, row['prefix47_s'], attempt + ': first 47 updates')
        audit.close(r['final_drain_s'], row['final_drain_s'], attempt + ': final drain')
        audit.equal(r['cell_id'] in groups[r['group_id']], False, attempt + ': unique group cell')
        groups[r['group_id']][r['cell_id']] = {'result': r, 'elapsed_s': elapsed, 'source_row': row['_source_row']}
        source_csv = 'data/original/allocation/runs.csv'
        outputs.append({
            'stage': 'allocation', 'design': 'cpu_allocation_' + r['family'], 'workload': 'UltraChat',
            'asset_block': 1, 'repeat_block': r['repeat_block'], 'position': r['position'],
            'condition': r['cell_id'], 'input_mode': 'online' if r['cell_id'].startswith('O') else 'pretokenized',
            'save_mode': 'async' if saving else 'none', 'save_interval_updates': 47 if saving else '',
            'slot_id': slot, 'elapsed_s': elapsed, 'supervised_tokens': TOKENS,
            'seconds_per_million_tokens': elapsed/(TOKENS/1e6), 'updates': 282,
            'periodic_commits': r['checkpoint_commits'], 'initial_commits': r['initial_checkpoint_commits'],
            'fetch_s': fetch, 'fetch_fraction': fetch/elapsed, 'replicate_scope': scope(r['family']),
            'source_csv': source_csv, 'source_row': row['_source_row'],
            'cost_source_csv': source_csv, 'cost_source_row': row['_source_row'],
            'original_source_path': 'analysis/runs.csv', 'original_cost_source_path': 'analysis/runs.csv',
            'attempt_id': attempt, 'placement': r['placement'], 'training_pool': r['training_pool'],
            'measurement_segment': r['measurement_segment'], 'boot_id': r['boot_id'],
            'crosses_user_pause': False, 'training_cpus': json.dumps(r['training_cpus']),
            'writer_cpus': json.dumps(r['writer_cpus']), 'source_result': path.relative_to(ROOT).as_posix(),
            'original_result_path': r['_provenance']['original_source_path'],
            'original_result_sha256': full_original_sha, 'pre_first_save_s': prefix,
            'final_drain_s': r['final_drain_s'], 'family': r['family'], 'group_id': r['group_id'],
            'layout': r['layout'], 'half_pool': r['half_pool'], 'session_id': r['session_id'],
            'comparable_implementation': r['comparable_implementation'], 'schedule_index': r['schedule_index'],
            'raw_start_ns': r['measurement_start_raw_ns'], 'raw_end_ns': r['measurement_end_raw_ns'],
        })
    audit.equal(len(starts), 1, 'common starting-state digest')
    audit.equal(len(scientific_protocols), 1, 'common scientific protocol')
    audit.equal({r['slot_id'] for r in outputs}, set(planned), 'all frozen slots covered')
    actual_order = [r['slot_id'] for r in sorted(attempts, key=lambda a:a['start_utc']) if r['adopted']=='True']
    frozen_order = [r['slot_id'] for r in sorted(order, key=lambda a:int(a['schedule_index']))]
    audit.equal(actual_order, frozen_order, 'adopted execution order')
    unused = [r for r in attempts if r['adopted']!='True']
    audit.equal([r['attempt_id'] for r in unused], ['alloc_B01_PS'], 'only interrupted attempt excluded')
    interrupted = read_json('results/alloc_B01_PS.interrupted.json')
    audit.equal(interrupted['valid'], False, 'interrupted result invalid')
    return sorted(outputs, key=lambda a:a['schedule_index']), groups, unused


def reconstruct_effects(groups, audit):
    refs = {r['group_id']: r for r in read_csv('group_effects.csv')}
    group_refs = {r['group_id']: r for r in read_csv('groups.csv')}
    output = []
    for group_id, cells in sorted(groups.items()):
        family = group_id[0]
        audit.equal(set(cells), set(CELLS[family]), group_id + ': complete four-cell matrix')
        records = [c['result'] for c in cells.values()]
        for key in ['session_id', 'boot_id', 'measurement_segment', 'comparable_implementation', 'half_pool']:
            audit.equal(len({r[key] for r in records}), 1, group_id + ': single ' + key)
        audit.equal(group_refs[group_id]['contrast_eligible'], 'True', group_id + ': reported eligibility')
        value = {cell: data['elapsed_s'] for cell, data in cells.items()}
        xo, yo, xp, yp = CELLS[family]
        ro, rp = value[xo]-value[yo], value[xp]-value[yp]
        interaction = ro-rp
        bx, by = value[xo]-value[xp], value[yo]-value[yp]
        audit.close(interaction, bx-by, group_id + ': difference-in-differences identity')
        for key, val in {'R_O':ro, 'R_P':rp, 'interaction':interaction, **value}.items():
            audit.close(val, refs[group_id][key], group_id + ': reported ' + key)
        r = records[0]
        x_setting, y_setting = SETTINGS[family]
        output.append({
            'stage':'allocation', 'design':'cpu_allocation_'+family, 'workload':'UltraChat',
            'block':group_id, 'replicate_scope':scope(family), 'tokens_per_run':TOKENS,
            # These two settings retain the existing CACHE-INTERACTION convention.
            'reference_setting':y_setting, 'target_setting':x_setting,
            'cache_savings_reference_s':by, 'cache_savings_target_s':bx,
            'cache_savings_reference_s_per_mtoken':by/(TOKENS/1e6),
            'cache_savings_target_s_per_mtoken':bx/(TOKENS/1e6),
            'cache_savings_reference_pct':100*by/value[yo], 'cache_savings_target_pct':100*bx/value[xo],
            'interaction_s':interaction, 'interaction_s_per_mtoken':interaction/(TOKENS/1e6),
            'interaction_original_name':INTERACTIONS[family], 'interaction_original_value':interaction,
            'interaction_original_unit':'seconds_per_fixed_U1_window',
            'interaction_normalized_pct':'', 'interaction_normalization':'not_reported',
            'reported_B_target_pct':'',
            'source_rows':';'.join(f"data/original/allocation/runs.csv:{cells[c]['source_row']}" for c in CELLS[family]),
            'reference_effect_csv':'data/original/allocation/group_effects.csv',
            'reference_effect_row':refs[group_id]['_source_row'],
            'family':family, 'group_id':group_id, 'half_pool':r['half_pool'], 'session_id':r['session_id'],
            'boot_id':r['boot_id'], 'measurement_segment':r['measurement_segment'],
            'comparable_implementation':r['comparable_implementation'], 'group_complete':True,
            'contrast_eligible':True, 'crosses_user_pause':False,
            # The DIRECT performance comparison instead goes from X to Y.
            'direct_reference_setting':x_setting, 'direct_target_setting':y_setting,
            'R_O':ro, 'R_P':rp, INTERACTIONS[family]:interaction,
            'R_O_pct_of_reference':100*ro/value[xo], 'R_P_pct_of_reference':100*rp/value[xp], **value,
        })
    audit.equal(len(output), 16, 'eligible allocation groups')
    for family, expected in GROUP_COUNTS.items():
        audit.equal(sum(e['family']==family for e in output), expected, family + ': group count')
    pairs = read_csv('available_pairs.csv')
    audit.equal(len(pairs), 32, 'available input pairs')
    lookup = {e['group_id']:e for e in output}
    for pair in pairs:
        effect = lookup[pair['group_id']]
        audit.equal(pair['from_eligible_complete_group'], 'True', pair['group_id'] + ': complete-group pair')
        audit.equal(pair['session_id'], effect['session_id'], pair['group_id'] + ': pair session')
        audit.close(float(pair['R']), effect['R_'+pair['input']], pair['group_id'] + ': direct pair')
    return output


def summarize(runs, effects, unused, manifest, audit):
    refs = read_json('summary.json')
    metrics, cells, percentages, strata = {}, {}, {}, {}
    for family in ['A','B','C']:
        fs = [e for e in effects if e['family']==family]
        metrics[family] = {}
        for metric in ['R_O','R_P','interaction']:
            field = 'interaction_s' if metric=='interaction' else metric
            calculated = descriptive(e[field] for e in fs)
            for key, val in calculated.items():
                audit.close(val, refs['metrics'][family][metric][key], family+'/'+metric+'/'+key)
            calculated['group_ids'] = [e['group_id'] for e in fs]
            metrics[family][metric] = calculated
        cells[family] = {c:{'n':len(fs), **{field:statistics.fmean(r[field] for r in runs if r['family']==family and r['condition']==c)
                                        for field in ['elapsed_s','fetch_s','pre_first_save_s','final_drain_s']}}
                         for c in CELLS[family]}
        percentages[family] = {}
        xo,yo,xp,yp = CELLS[family]
        for label,x,y,metric in [('O',xo,yo,'R_O'),('P',xp,yp,'R_P')]:
            base,target = cells[family][x]['elapsed_s'],cells[family][y]['elapsed_s']
            percentages[family][label] = {
                'reference_cell':x,'target_cell':y,'reference_mean_s':base,'target_mean_s':target,
                'mean_elapsed_reduction_s':base-target,
                'elapsed_reduction_pct_of_means':100*(base-target)/base,
                'mean_group_elapsed_reduction_pct':statistics.fmean(e[metric+'_pct_of_reference'] for e in fs),
            }
        for pool in ['P','Q']:
            name=family+'_pool_'+pool
            calculated=descriptive(e['interaction_s'] for e in fs if e['half_pool']==pool)
            for key,val in calculated.items():
                audit.close(val,refs['strata'][name][key],name+'/'+key)
            strata[name]=calculated
    totals={k:sum(r[k] for r in runs) for k in ['updates','supervised_tokens','periodic_commits','initial_commits']}
    audit.equal(totals,{'updates':18048,'supervised_tokens':254459776,'periodic_commits':288,'initial_commits':48},'allocation work totals')
    session_counts=Counter((r['session_id'],r['boot_id']) for r in runs)
    return {
        'stage':'allocation','run_count':len(runs),'complete_groups':len(effects),'eligible_groups':len(effects),
        'available_pairs':32,'partial_pairs':0,'actual_attempts':65,'unused_attempts':unused,
        'family_run_counts':dict(Counter(r['family'] for r in runs)),
        'family_group_counts':dict(Counter(e['family'] for e in effects)),
        'interaction_names':INTERACTIONS, **totals,'microbatches':totals['updates']*8,
        'metrics':metrics,'cell_means_by_family':cells,'direct_percentages':percentages,'strata':strata,
        'session_boot_counts':[{'session_id':key[0],'boot_id':key[1],'valid_runs':count} for key,count in sorted(session_counts.items())],
        'percentage_definition':'Elapsed reduction = 100*(mean reference elapsed - mean target elapsed)/mean reference elapsed. Positive means the direct-comparison target is faster. The separate mean_group value averages individual group percentages; neither is throughput speedup.',
        'effect_column_convention':'Common reference_setting/target_setting describe CACHE benefits: interaction = cache_savings_target - cache_savings_reference. Direct-reference and direct-target fields instead describe the X-to-Y layout change and R=T_X-T_Y.',
        'ci_qualification':'Small-sample two-sided t approximations across same-U1 time groups. Independence and exchangeability under temporal drift are not established. An interval crossing zero is not evidence of no effect.',
        'loo_qualification':'Each entry removes the corresponding group_ids entry; all eligible groups remain in the primary estimate.',
        'scope_qualification':'A/B/C are distinct four-condition comparisons, never pooled. All groups are internally one session and boot; A01 is in session_001 and all other groups in session_002. Frozen-field arithmetic does not independently reproduce training or full telemetry.',
        'verification':{'arithmetic_comparisons':len(audit.comparisons),'identity_checks':audit.identities,
                        'max_absolute_error':max(c['absolute_error'] for c in audit.comparisons),
                        'source_files_checked':len(manifest),'all_packaged_source_hashes_matched':True,
                        'first_valid_selection_verified':True,'frozen_order_preserved':True,
                        'all_groups_single_session_boot':True},
    }


def atomic_bytes(path, content):
    temp=path.with_suffix(path.suffix+'.tmp')
    with temp.open('wb') as f:
        f.write(content);f.flush();os.fsync(f.fileno())
    os.replace(temp,path)


def make_figures(effects):
    os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'case_study_allocation_matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5,'axes.titlesize':9,
                         'pdf.fonttype':42,'ps.fonttype':42})
    titles={'A':'A  Full16 → Split8+8','B':'B  Shared8 → Split8+8','C':'C  No saving: 16 → 8'}
    colors={'R_O':'#2166a5','R_P':'#c56b18'}
    markers={'R_O':'o','R_P':'s'}
    figures=ROOT/'figures';figures.mkdir(exist_ok=True)
    for mode in ['direct_effects','interactions']:
        fig,axes=plt.subplots(1,3,figsize=(8.4,3.15),sharey=True)
        for ax,family in zip(axes,['A','B','C']):
            fs=[e for e in effects if e['family']==family]
            n=len(fs)
            ax.axhline(0,color='#424242',linewidth=.9,zorder=1)
            if mode=='direct_effects':
                for metric,offset in [('R_O',-.12),('R_P',.12)]:
                    mean=statistics.fmean(e[metric] for e in fs)
                    ax.axhline(mean,color=colors[metric],linestyle='--',linewidth=.9,alpha=.9,zorder=1)
                    for i,e in enumerate(fs,1):
                        ax.scatter(i+offset,e[metric],s=26,marker=markers[metric],
                                   facecolors='white' if e['group_id']=='A01' else colors[metric],
                                   edgecolors=colors[metric],linewidths=1.1,zorder=3)
                ax.set_ylim(-45,40)
            else:
                values=[e['interaction_s'] for e in fs]
                mean=statistics.fmean(values)
                ax.axhline(mean,color='#2166a5',linestyle='--',linewidth=1,
                           label=f'${INTERACTIONS[family]}$ mean {mean:+.2f} s',zorder=1)
                for i,e in enumerate(fs,1):
                    ax.scatter(i,e['interaction_s'],s=27,marker='o',
                               facecolors='white' if e['group_id']=='A01' else '#2166a5',
                               edgecolors='#2166a5',linewidths=1.1,zorder=3)
                ax.set_ylim(-50,55)
                ax.legend(loc='upper left',frameon=False,fontsize=7.5,handlelength=1.5)
            ax.set_title(titles[family],loc='left',pad=9)
            ax.set_xlim(.55,n+.45)
            ax.set_xticks(range(1,n+1),[e['group_id']+('*' if e['group_id']=='A01' else '') for e in fs],fontsize=7)
            ax.set_xlabel('Temporal group',labelpad=5)
            ax.grid(axis='y',color='#e2e2e2',linewidth=.5,zorder=0)
            ax.spines[['top','right']].set_visible(False)
        axes[0].set_ylabel(('Direct reduction' if mode=='direct_effects' else 'Interaction')+' (s / fixed U1)')
        if mode=='direct_effects':
            handles=[Line2D([],[],color=colors['R_O'],marker='o',linestyle='None',label='Online input ($R_O$)'),
                     Line2D([],[],color=colors['R_P'],marker='s',linestyle='None',label='Pretokenized input ($R_P$)'),
                     Line2D([],[],color='#555555',linestyle='--',label='Within-family mean')]
            fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.54,1.01),ncol=3,frameon=False,fontsize=8)
            foot='Positive direct reduction means the configuration after the arrow is faster.'
        else:
            foot='Interaction = $R_O - R_P$. A positive value does not imply that both input paths become faster.'
        fig.text(.5,.105,foot,ha='center',va='center',fontsize=7.5)
        fig.text(.5,.045,'* A01 finished before the pause; every group uses a single session and boot. No effects are pooled.',
                 ha='center',va='center',fontsize=7.2)
        fig.subplots_adjust(left=.085,right=.985,bottom=.28,top=.81,wspace=.16)
        for extension in ['png','pdf']:
            buffer=io.BytesIO();fig.savefig(buffer,format=extension,dpi=300,facecolor='white')
            path=figures/f'allocation_{mode}.{extension}'
            atomic_bytes(path,buffer.getvalue())
            if path.stat().st_size<1000:
                raise ValueError(f'Empty/incomplete figure: {path}')
        plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skip-figure',action='store_true')
    args=parser.parse_args()
    audit=Audit()
    manifest=verify_sources(audit)
    runs,groups,unused=reconstruct_runs(audit)
    effects=reconstruct_effects(groups,audit)
    summary=summarize(runs,effects,unused,manifest,audit)
    write_csv(DATA/'allocation_runs.csv',MAIN_COLUMNS+RUN_EXTRA,runs)
    write_csv(DATA/'allocation_effects.csv',EFFECT_COLUMNS+EFFECT_EXTRA,effects)
    (DATA/'allocation_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    if not args.skip_figure:
        make_figures(effects)
    print(json.dumps({'runs':len(runs),'effects':len(effects),'updates':summary['updates'],
                      'supervised_tokens':summary['supervised_tokens'],**summary['verification'],
                      'figures_generated':not args.skip_figure},indent=2))


if __name__=='__main__':
    main()
