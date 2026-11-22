#!/usr/bin/env python3
"""Reconstruct five case study rounds without running training or pooling effects."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'


def read_table(name):
    with (DATA / name).open(encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        return list(reader.fieldnames), rows


def combine(inputs, output):
    fields, rows = [], []
    for source in inputs:
        source_fields, source_rows = read_table(source)
        fields.extend(x for x in source_fields if x not in fields)
        rows.extend(source_rows)
    with (DATA / output).open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skip-figure', action='store_true')
    args = parser.parse_args()
    module_outputs = {}
    for script in ['reproduce_prior.py', 'reproduce_placement.py', 'reproduce_allocation.py']:
        command = [sys.executable, str(ROOT / 'analysis' / script)]
        if args.skip_figure:
            command.append('--skip-figure')
        process = subprocess.run(command, text=True, capture_output=True)
        if process.returncode:
            sys.stderr.write(process.stdout + process.stderr)
            raise SystemExit(process.returncode)
        module_outputs[script] = process.stdout.strip()
    runs = combine(['prior_main_runs.csv', 'placement_runs.csv', 'allocation_runs.csv'], 'main_runs.csv')
    effects = combine(['prior_effects.csv', 'placement_effects.csv', 'allocation_effects.csv'], 'effects.csv')
    if len(runs) != 156 or len(effects) != 36:
        raise ValueError('Expected 156 completed runs and 36 within-design effect rows')
    if len({(x['stage'], x['slot_id']) for x in runs}) != 156:
        raise ValueError('Repeated stage/slot identifiers')
    if len({(x['stage'], x['design'], x['workload'], x['block']) for x in effects}) != 36:
        raise ValueError('Repeated stage/design/workload/block effect identifiers')
    totals = {key: sum(int(x[key]) for x in runs)
              for key in ['updates', 'supervised_tokens', 'periodic_commits', 'initial_commits']}
    expected = {'updates': 45504, 'supervised_tokens': 587858856,
                'periodic_commits': 672, 'initial_commits': 116}
    if totals != expected:
        raise ValueError(f'Work totals changed: {totals}')
    prior = json.loads((DATA / 'prior_summary.json').read_text())
    placement = json.loads((DATA / 'placement_summary.json').read_text())
    allocation = json.loads((DATA / 'allocation_summary.json').read_text())
    placement_sources = json.loads((DATA / 'placement_source_manifest.json').read_text())
    summary = {
        'main_runs': len(runs), 'effect_rows': len(effects), **totals,
        'microbatches': totals['updates'] * 8,
        'run_counts_by_stage': dict(Counter(x['stage'] for x in runs)),
        'effect_counts_by_stage': dict(Counter(x['stage'] for x in effects)),
        'run_counts_by_design': dict(Counter(x['design'] for x in runs)),
        'effect_counts_by_design': dict(Counter(x['design'] for x in effects)),
        'scope': 'Five experimental rounds, including three allocation comparison families; rows are combined for audit, effects are not pooled.',
        'statistical_unit': 'Complete within-design, within-workload block; not runs, updates or saves.',
        'placement_block3': 'Retained; only OS before user pause/WSL reboot, other five cells after.',
        'allocation_sessions': 'All 16 groups use one session and boot internally. A01 is entirely in session_001; all other groups are in session_002. The interrupted B01_PS is excluded and its first valid retry is selected.',
        'effect_units': {
            'finish_v3/G': 'percent of online no-save time',
            'frequency/I': 'seconds per million supervised tokens',
            'overlap/H': 'seconds per million supervised tokens',
            'placement/J': 'seconds per fixed U1 training window',
            'allocation/A/K_A': 'seconds per fixed U1 training window',
            'allocation/B/J_B': 'seconds per fixed U1 training window',
            'allocation/C/K_C': 'seconds per fixed U1 training window',
        },
        'prior_designs': prior,
        'placement': placement,
        'allocation': allocation,
        'verification': {
            'prior_comparisons': prior['verification']['comparisons'],
            'placement_comparisons': placement['verification']['arithmetic_comparisons'],
            'allocation_comparisons': allocation['verification']['arithmetic_comparisons'],
            'source_files_checked': (prior['verification']['source_hashes_checked'] +
                                     len(placement_sources) + allocation['verification']['source_files_checked']),
            'module_outputs': module_outputs,
            'scope': 'Offline arithmetic and packaged source hashes, not independent training reproduction.',
        },
    }
    (DATA / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'main_runs': len(runs), 'effect_rows': len(effects), **totals,
                      'run_counts_by_stage': summary['run_counts_by_stage'],
                      'figures_generated': not args.skip_figure}, indent=2))


if __name__ == '__main__':
    main()
