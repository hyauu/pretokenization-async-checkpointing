import json,sys,time,pathlib
boot=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()
for line in sys.stdin:
 print(json.dumps({'raw_ns':time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW),'boot_id':boot,'echo':line.strip()}),flush=True)
