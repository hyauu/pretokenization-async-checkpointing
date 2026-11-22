import bisect,ctypes,datetime,json,os,pathlib,statistics,subprocess,threading,time,sys
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'host_deps'))
import psutil
R=pathlib.Path(__file__).resolve().parents[1]
DISTRO='S09Replication-20261027';G='/opt/s09_replication_20261027';GH='/mnt/c/Users/anonymous/Documents/ChatGPT/S09Replication_2026-10-27'
Q=ctypes.c_longlong();ctypes.windll.kernel32.QueryPerformanceFrequency(ctypes.byref(Q));F=Q.value
def qpc():
 v=ctypes.c_longlong();ctypes.windll.kernel32.QueryPerformanceCounter(ctypes.byref(v));return v.value
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(path,obj):
 path=pathlib.Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2),encoding='utf-8');os.replace(tmp,path)
def append(path,obj):
 with pathlib.Path(path).open('a',encoding='utf-8') as f:f.write(json.dumps(obj)+'\n')
def guest(args,**kw):return subprocess.run(['wsl','-d',DISTRO,'-u','root','--exec',*args],**kw)
class Monitoring:
 def __init__(self):self.stop=threading.Event();self.points=[];self.gpu=[];self.errors=[];self.children=[];self.threads=[]
 def start(self):
  for method in (self.clock,self.gpu_sample,self.host_sample):
   t=threading.Thread(target=method,daemon=True);t.start();self.threads.append(t)
 def clock(self):
  try:
   p=subprocess.Popen(['wsl','-d',DISTRO,'-u','root','--exec','/opt/s09/venv/bin/python','-u',G+'/src/clock_guest.py'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=(R/'host/clock.stderr.log').open('ab'),text=True,bufsize=1);self.children.append(p)
   while not self.stop.is_set():
    before=qpc();p.stdin.write(str(before)+'\n');p.stdin.flush();v=json.loads(p.stdout.readline());after=qpc()
    v.update(qpc_before=before,qpc_after=after,qpc_mid=(before+after)/2,qpc_frequency=F,utc=utc(),rtt_s=(after-before)/F)
    self.points.append(v);append(R/'evidence/clock_bridge.jsonl',v)
    save(R/'host/clock_state.json',dict(status='RUNNING',pid=p.pid,**v));self.stop.wait(1)
  except Exception as e:self.errors.append('clock: '+repr(e))
 def gpu_sample(self):
  fields=['pstate','utilization.gpu','utilization.memory','memory.used','clocks.sm','clocks.mem','power.draw','power.limit','temperature.gpu','clocks_event_reasons.active','clocks_event_reasons.sw_power_cap','clocks_event_reasons.hw_slowdown','clocks_event_reasons.hw_thermal_slowdown','clocks_event_reasons.hw_power_brake_slowdown','clocks_event_reasons.sw_thermal_slowdown','clocks_event_reasons.board_limit','clocks_event_reasons.reliability']
  save(R/'evidence/gpu_schema.json',dict(fields=fields,period_s=1,interpretation='Timestamp is host receipt. Optional fields unavailable are NA. Clip to mapped measured windows.'))
  try:
   p=subprocess.Popen(['nvidia-smi','--query-gpu='+','.join(fields),'--format=csv,noheader,nounits','-lms','1000'],stdout=subprocess.PIPE,stderr=(R/'host/gpu.stderr.log').open('ab'),text=True,bufsize=1);self.children.append(p)
   for line in p.stdout:
    if self.stop.is_set():break
    values=[v.strip() for v in line.strip().split(',')]
    if len(values)!=len(fields):self.errors.append('GPU query: '+line[:200]);break
    row=dict(qpc=qpc(),qpc_frequency=F,utc=utc(),**dict(zip(fields,values)));self.gpu.append(row);append(R/'evidence/gpu.jsonl',row)
  except Exception as e:self.errors.append('GPU: '+repr(e))
 def host_sample(self):
  while not self.stop.is_set():
   row=dict(qpc=qpc(),qpc_frequency=F,utc=utc(),cpu_times=psutil.cpu_times()._asdict(),memory=psutil.virtual_memory()._asdict(),swap=psutil.swap_memory()._asdict(),disk={k:v._asdict() for k,v in (psutil.disk_io_counters(perdisk=True) or {}).items()})
   processes=[]
   for p in psutil.process_iter(['pid','name','cpu_times','memory_info']):
    try:
     t=p.info['cpu_times'];mi=p.info['memory_info'];processes.append(dict(pid=p.pid,name=p.info['name'],cpu_s=t.user+t.system,rss=mi.rss))
    except (psutil.NoSuchProcess,psutil.AccessDenied,AttributeError):pass
   row['processes']=processes;append(R/'evidence/host_resources.jsonl',row);self.stop.wait(5)
 def qualify(self):
  pts=[p for p in self.points if p['rtt_s']<.05]
  assert len(pts)>=45,'Insufficient low-latency clock samples'
  a,b=pts[0],pts[-1];raw=(b['raw_ns']-a['raw_ns'])/1e9;host=(b['qpc_mid']-a['qpc_mid'])/F
  assert raw>=45 and abs(raw-host)<=.05 and a['boot_id']==b['boot_id'],(raw,host)
  result=dict(status='PASS',samples=len(pts),span_raw_s=raw,span_host_s=host,difference_s=raw-host,max_rtt_s=max(p['rtt_s'] for p in pts),boot_id=a['boot_id'])
  save(R/'evidence/clock_qualification.json',result);return result
 def validate(self,start,end,boot):
  pts=[p for p in self.points if p['boot_id']==boot and p['rtt_s']<.05]
  xs=[p['raw_ns'] for p in pts]
  def mapped(x):
   i=bisect.bisect_right(xs,x);assert 0<i<len(xs),'Clock bridge lacks endpoint coverage'
   a,b=pts[i-1],pts[i];assert (b['raw_ns']-a['raw_ns'])/1e9<5,'Clock gap'
   f=(x-a['raw_ns'])/(b['raw_ns']-a['raw_ns']);return a['qpc_mid']+f*(b['qpc_mid']-a['qpc_mid'])
  qs,qe=mapped(start),mapped(end);raw=(end-start)/1e9;host=(qe-qs)/F
  covering=[p for p in pts if start<=p['raw_ns']<=end]
  assert len(covering)>raw/3 and max((b['raw_ns']-a['raw_ns'])/1e9 for a,b in zip(covering,covering[1:]))<5
  assert abs(raw-host)<=max(.05,raw*.0002),(raw,host)
  return dict(status='PASS',start_qpc=qs,end_qpc=qe,raw_s=raw,host_s=host,difference_s=raw-host,bridge_points=len(covering),boot_id=boot)
 def close(self):
  self.stop.set()
  for p in self.children:
   try:
    if p.poll() is None:p.terminate()
   except OSError:pass
  for p in self.children:
   try:p.wait(timeout=10)
   except subprocess.TimeoutExpired:p.kill();p.wait()
  for t in self.threads:t.join(timeout=10)
