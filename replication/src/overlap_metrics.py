"""Pure CPU RAW interval diagnostics; update spans are not GPU kernel spans."""
CELLS=('O_ASYNC','P_ASYNC','O_WAIT','P_WAIT');SCALE=3.975934
def union(intervals):
    out=[]
    for a,b in sorted(intervals):
        if b<=a:continue
        if out and a<=out[-1][1]:out[-1]=(out[-1][0],max(out[-1][1],b))
        else:out.append((a,b))
    return out
def intersection_seconds(a,b,intervals):return sum(max(0,min(b,y)-max(a,x)) for x,y in union(intervals))
def summarize(vals):
    def difference(a,b):return vals[a]-vals[b] if a in vals and b in vals else None
    da=difference('O_ASYNC','P_ASYNC');dw=difference('O_WAIT','P_WAIT');h=da-dw if da is not None and dw is not None else None;s=sum(vals.values())/4 if len(vals)==4 else None
    return dict(D_async=da,D_wait=dw,H=h,S=s,Q=100*h/s if h is not None else None,B_async=-100*da/vals['O_ASYNC'] if da is not None else None,B_wait=-100*dw/vals['O_WAIT'] if dw is not None else None,R_O=difference('O_WAIT','O_ASYNC'),R_P=difference('P_WAIT','P_ASYNC'),H_seconds=h*SCALE if h is not None else None)
def prefix47(v):
    rr=v['update_records'];assert len(rr)>=47
    pre=rr[46]['end']-v['measurement_start_raw_ns']/1e9
    assert sum(x['tokens'] for x in rr[:47])==672080 and sum(x['tokens'] for x in rr[47:])==3303854
    return dict(pre47_elapsed_s=pre,post_elapsed_s=v['elapsed_s']-pre,pre47_tokens=672080,post_tokens=3303854)
def overlap_events(v):
    records=v['update_records'];events=v['save_events'];out=[]
    for u in (47,94,141,188,235):
        step=3+u;call=next(e for e in events if e['event']=='save_call' and e['step']==step and not e['initial']);commit=next(e for e in events if e['event']=='commit' and e['step']==step and not e['initial'])
        following=records[u:];overlap=intersection_seconds(call['start'],commit['end'],[(x['start'],x['end']) for x in following]);involved=[x['measurement_update'] for x in following if min(commit['end'],x['end'])>max(call['start'],x['start'])]
        out.append(dict(measurement_update=u,global_step=step,save_start_raw_s=call['start'],save_call_end_raw_s=call['end'],commit_end_raw_s=commit['end'],inflight_s=commit['end']-call['start'],overlap_s=overlap,next_update_start_raw_s=following[0]['start'],uncommitted_at_next_update=commit['end']>following[0]['start'],involved_update_count=len(involved),involved_measurement_updates=involved,post_save_drain_end_raw_s=records[u-1].get('post_save_drain_end_raw_s')))
    return out
