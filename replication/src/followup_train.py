from timing import raw_ns, raw_seconds
"""One fresh GPU process: calibration, common anchor, correctness, or frozen performance."""
import argparse
import copy
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist

from common import (ROOT, WORKLOAD, dump, filehash, init_pg, build_model, make_loader,
                    rng_state, restore_rng, tree_digest, batch_digest)
from checkpoint import Saver, load_path
from train_core import update, require_exhausted, persistent_buffer_keys


def cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_copy(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_copy(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_copy(v) for v in value)
    return copy.deepcopy(value)


def summary(model, optimizer, loader, mf, step, cursor):
    parts = {'model': tree_digest(model.state_dict()), 'optimizer': tree_digest(optimizer.state_dict()),
             'rng': tree_digest(rng_state()), 'loader_generator': tree_digest(loader.generator.get_state()),
             'manifest_sha256': mf['manifest_sha256'], 'step': step, 'cursor': cursor,
             'accumulation_boundary': 0}
    return {**parts, 'complete_digest': tree_digest(parts)}


def save_anchor(path, model, optimizer, loader, mf, step, cursor):
    path = Path(path)
    if path.exists():
        raise RuntimeError('Anchor overwrite prohibited')
    payload = {'model': cpu_copy(model.state_dict()), 'optimizer': cpu_copy(optimizer.state_dict()),
               'rng': cpu_copy(rng_state()), 'loader_generator': loader.generator.get_state().clone(),
               'manifest_sha256': mf['manifest_sha256'], 'global_step': step,
               'cursor': cursor, 'accumulation_boundary': 0, 'workload': WORKLOAD}
    expected = summary(model, optimizer, loader, mf, step, cursor)
    tmp = path.with_suffix('.tmp')
    with tmp.open('xb') as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)
    expected.update(file_sha256=filehash(path), bytes=path.stat().st_size, path=str(path))
    dump(path.with_suffix('.json'), expected)
    return expected


def restore_anchor(path, model, optimizer, loader, mf):
    payload = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    assert payload['manifest_sha256'] == mf['manifest_sha256']
    assert payload['workload'] == WORKLOAD and payload['accumulation_boundary'] == 0
    model.load_state_dict(payload['model'], strict=True)
    optimizer.load_state_dict(payload['optimizer'])
    optimizer.zero_grad(set_to_none=True)
    restore_rng(payload['rng'])
    loader.generator.set_state(payload['loader_generator'])
    return {'rng': cpu_copy(payload['rng']), 'loader_generator': payload['loader_generator'].clone(),
            'step': payload['global_step'], 'cursor': payload['cursor']}


def strip_events(records):
    for record in records:
        events = record.pop('cuda_events', None)
        if events is not None:
            record['gpu_event_ms'] = events[0].elapsed_time(events[1])
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--request', required=True)
    request = json.loads(Path(ap.parse_args().request).read_text())
    attempt = request['attempt_id']
    assert request['round_id'] == 's09_replication_20261027'
    assert request['workload'] == WORKLOAD
    out = ROOT / 'runs' / attempt
    out.mkdir(exist_ok=True, parents=True)
    dump(out / 'request.json', request)
    action = request['action']
    condition = request.get('condition', 'O0')
    save_schedule = request['save_schedule']
    assert condition in ('O0', 'OA', 'P0', 'PA')
    mf = json.loads(Path(request['manifest']).read_text())
    protocol_sha = 'pre_freeze_correctness'
    if request.get('protocol'):
        protocol_sha = filehash(request['protocol'])
        protocol = json.loads(Path(request['protocol']).read_text())
        for name, digest in protocol['source_sha256'].items():
            assert filehash(ROOT / 'src' / name) == digest, ('Source changed after freeze', name)
        from specification import validate_request
        spec = validate_request(request, protocol, mf)
        assert protocol['primary_clock']=='CLOCK_MONOTONIC_RAW'
        frozen_anchor=protocol['anchors'][f"{WORKLOAD}_{request['block']}"]
        live_anchor=json.loads(Path(request['anchor']).with_suffix('.json').read_text())
        assert frozen_anchor==live_anchor,'Anchor metadata changed after freeze'
    assert mf['microbatch'] == 2 and mf['accumulation'] == 8 and mf['warmup_updates'] == 3
    torch.set_num_threads(8)
    torch.set_num_interop_threads(16)
    init_pg()
    t_load = raw_seconds()
    model, optimizer, loss = build_model(mf['seed'])
    assert not persistent_buffer_keys(model)
    loader, sampler = make_loader('cached' if condition.startswith('P') else 'online', 0, mf['warmup'], mf['seed'])
    load_s = raw_seconds() - t_load
    accum = mf['accumulation']
    warm_records = []
    if action != 'verify_anchor' and action != 'verify_snapshot':
        t_warm = raw_seconds()
        iterator = iter(loader)
        for i in range(3):
            warm_records.append(update(model, optimizer, loss, iterator, mf['warmup'][i*accum:(i+1)*accum], accum))
        require_exhausted(iterator)
        torch.cuda.synchronize()
        warm_s = raw_seconds() - t_warm
    else:
        warm_s = 0
    cursor = len(mf['warmup'])
    step = 3
    base = {**{k:request[k] for k in ('group_id','family','layout','half_pool','schedule_index','position','session_id','measurement_segment','comparable_implementation','input_dose')}, 'action': action, 'attempt_id': attempt, 'round_id': request['round_id'],
            'workload': WORKLOAD, 'condition': condition, 'save_schedule': save_schedule, 'cell_id': request['cell_id'], 'repeat_block': request['repeat_block'], 'interval': request['interval'], 'execution_protocol_sha256': request['execution_protocol_sha256'], 'manifest_sha256': mf['manifest_sha256'],
            'protocol_sha256': protocol_sha, 'load_s': load_s, 'warmup_s': warm_s,
            'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            'torch_cpu_threads': torch.get_num_threads(), 'torch_interop_threads': torch.get_num_interop_threads(),
            'primary_clock':'CLOCK_MONOTONIC_RAW','auxiliary_perf_ns_at_setup':time.perf_counter_ns()}
    if action == 'anchor':
        base.update(anchor=save_anchor(request['anchor'], model, optimizer, loader, mf, step, cursor),
                    status='ANCHOR_CREATED', valid=True)
        dump(out / 'result.json', base)
        dist.destroy_process_group()
        return
    if action == 'verify_snapshot':
        ref = json.loads(Path(request['reference']).read_text())
        meta = load_path(request['checkpoint'], model, optimizer, mf['manifest_sha256'])
        loader.generator.set_state(meta['extra']['loader_generator_state'])
        observed = summary(model, optimizer, loader, mf, meta['global_step'], meta['committed_batch_cursor'])
        assert observed == ref['summary'], (observed, ref['summary'])
        sampler.entries = mf['measurement'][meta['committed_batch_cursor']-len(mf['warmup']):]
        batch = next(iter(loader))
        assert batch_digest(batch) == ref['next_batch_digest']
        base.update(status='SNAPSHOT_EXACT_LOAD_PASS', valid=True, observed=observed,
                    next_batch_digest=batch_digest(batch), loaded_step=meta['global_step'])
        dump(out / 'result.json', base)
        dist.destroy_process_group()
        return
    anchor_state = None
    t_restore = raw_seconds()
    if action != 'calibrate':
        anchor_state = restore_anchor(request['anchor'], model, optimizer, loader, mf)
        assert anchor_state['step'] == step and anchor_state['cursor'] == cursor
    restore_s = raw_seconds() - t_restore
    if action == 'verify_anchor':
        observed = summary(model, optimizer, loader, mf, step, cursor)
        expected = json.loads(Path(request['anchor']).with_suffix('.json').read_text())
        assert all(observed[k] == expected[k] for k in observed)
        base.update(status='ANCHOR_EXACT_LOAD_PASS', valid=True, observed=observed)
        dump(out / 'result.json', base)
        dist.destroy_process_group()
        return
    saver = Saver(attempt, 1, protocol_sha,
                  extra_metadata=lambda: {'loader_generator_state': loader.generator.get_state().clone()}) if condition.endswith('A') else None
    initial_save_start = raw_seconds()
    if saver:
        saver.save(model, optimizer, step, cursor, mf['manifest_sha256'], initial=True)
        saver.drain()
    initial_save_s = raw_seconds() - initial_save_start
    from placement import prepare, endpoint
    placement_info = prepare(request, out, saver is not None)
    if anchor_state:
        restore_rng(anchor_state['rng'])
        loader.generator.set_state(anchor_state['loader_generator'])
    t_hash = raw_seconds()
    initial = summary(model, optimizer, loader, mf, step, cursor) if anchor_state else None
    hash_s = raw_seconds() - t_hash
    if anchor_state:
        expected = json.loads(Path(request['anchor']).with_suffix('.json').read_text())
        assert all(initial[k] == expected[k] for k in initial)
    sampler.entries = mf['measurement']
    records = []
    actual_tokens = 0
    updates = mf['measurement_updates']
    interval = request.get('interval', 0)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    if action == 'snapshot':
        assert saver is not None
        iterator = iter(loader)
        first = update(model, optimizer, loss, iterator, mf['measurement'][:accum], accum, saver)
        step += 1
        cursor += accum
        captured = summary(model, optimizer, loader, mf, step, cursor)
        # Read the next batch using a separate CPU-only loader; restore training RNG afterward.
        saved_rng = rng_state()
        check_loader, _ = make_loader('online', 0, mf['measurement'][accum:2*accum], mf['seed'])
        next_digest = batch_digest(next(iter(check_loader)))
        restore_rng(saved_rng)
        path = saver.save(model, optimizer, step, cursor, mf['manifest_sha256'])
        pending_before_compute = not saver.staging.done()
        compute_start = raw_seconds()
        submit_observation={}
        def on_submit():
            submit_observation.update(first_forward_submit_raw_s=raw_seconds(),staging_future_pending=not saver.staging.done())
        second = update(model, optimizer, loss, iterator, mf['measurement'][accum:2*accum], accum, saver,on_first_forward=on_submit)
        saver.drain()
        stage_done = [e for e in saver.events if e['event'] == 'staging_done' and e['step'] == step]
        reference = {'summary': captured, 'next_batch_digest': next_digest, 'checkpoint': str(path)}
        dump(out / 'snapshot_reference.json', reference)
        base.update(status='SNAPSHOT_CAPTURED', valid=True, reference=reference,
                    staging_pending_at_next_update_entry=pending_before_compute,
                    next_update_entry_before_staging_callback=bool(stage_done and compute_start < stage_done[-1]['time']),
                    first_forward_submission_observation=submit_observation,
                    overlap_interpretation='Host first-forward submission vs staging future; this is not a GPU hardware execution timestamp.',
                    snapshot_global_step=step, save_events=saver.events)
        saver.close()
        dump(out / 'result.json', base)
        dist.destroy_process_group()
        return
    placement_pre = endpoint(placement_info, 'pre', out)
    start_ns = raw_ns()
    start = start_ns/1e9
    iterator = iter(loader)
    for i in range(updates):
        record = update(model, optimizer, loss, iterator, mf['measurement'][i*accum:(i+1)*accum], accum, saver, events=True)
        cursor += accum
        step += 1
        actual_tokens += record['tokens']
        record.update(global_step=step, measurement_update=i+1, framework_step_s=record['end']-record['start'])
        save_started = raw_seconds()
        periodic_save = saver is not None and (i+1) % interval == 0
        if periodic_save:
            saver.save(model, optimizer, step, cursor, mf['manifest_sha256'])
        record['save_call_s'] = raw_seconds()-save_started
        record['save_call_start_raw_s'] = save_started
        record['save_call_end_raw_s'] = save_started + record['save_call_s']
        record['post_save_drain_s'] = 0.0
        record['post_save_drain_start_raw_s'] = None
        record['post_save_drain_end_raw_s'] = None
        if periodic_save and save_schedule == 'serialize':
            wait_start = raw_seconds()
            saver.drain()
            wait_end = raw_seconds()
            record['post_save_drain_start_raw_s'] = wait_start
            record['post_save_drain_end_raw_s'] = wait_end
            record['post_save_drain_s'] = wait_end - wait_start
        records.append(record)
        print(json.dumps({'update': i+1, 'of': updates, 'loss': record['loss']}), flush=True)
    require_exhausted(iterator)
    torch.cuda.synchronize()
    before_drain = raw_seconds()
    if saver:
        saver.drain()
    torch.cuda.synchronize()
    end_ns = raw_ns()
    end = end_ns/1e9
    placement_post = endpoint(placement_info, 'post', out)
    assert actual_tokens == mf['effective_supervised_tokens']
    if saver:
        assert saver.commits == 1+updates//interval
    base.update(status='TRAINING_COMPLETED_PENDING_HOST_TIMING', valid=True,
                training_valid=True, timing_valid=None, placement_valid=None,
                placement=request['placement'], training_pool=request['training_pool'],
                training_cpus=request['training_cpus'], writer_cpus=request['writer_cpus'],
                placement_info=placement_info, placement_pre=placement_pre, placement_post=placement_post,
                pre_first_save_s=records[46]['end']-start,
                seed=mf['seed'], block=request.get('block'), completed_updates=updates,
                effective_supervised_tokens=actual_tokens, elapsed_s=end-start,
                measurement_start_raw_ns=start_ns, measurement_end_raw_ns=end_ns,
                measurement_start_raw_s=start, measurement_end_raw_s=end,
                elapsed_without_final_drain_s=before_drain-start, final_drain_s=end-before_drain,
                restore_anchor_s=restore_s, initial_save_s=initial_save_s,
                state_fingerprint_s=hash_s, measurement_initial_state=initial,
                anchor_path=request.get('anchor'),
                checkpoint_commits=saver.commits-1 if saver else 0,
                initial_checkpoint_commits=1 if saver else 0,
                final_global_step=step, final_consumed_cursor=cursor,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                update_records=strip_events(records), warmup_records=warm_records,
                save_events=saver.events if saver else [])
    dump(out / 'result.json', base)
    if saver:
        saver.close()
    dist.destroy_process_group()
    print(json.dumps({k:v for k,v in base.items() if k not in ('update_records','warmup_records','save_events')}), flush=True)


if __name__ == '__main__':
    main()
