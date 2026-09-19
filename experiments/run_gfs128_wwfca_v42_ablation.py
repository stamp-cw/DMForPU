"""Train timestep-only and timestep+noise WWFCA-v4.2 variants sequentially."""
from __future__ import annotations
import json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"experiments/results/gfs128_wwfca_v42"
METHODS=("wwfca_v42_time","wwfca_v42")
def status(value):
    OUT.mkdir(parents=True,exist_ok=True)
    target=OUT/"schedule_status.json";tmp=OUT/f"schedule_status.{os.getpid()}.tmp"
    tmp.write_text(json.dumps(value,indent=2),encoding="utf-8")
    for attempt in range(10):
        try:tmp.replace(target);return
        except PermissionError:
            if attempt==9:raise
            time.sleep(0.1*(attempt+1))
def main():
    OUT.mkdir(parents=True,exist_ok=True);started=time.time();done={}
    for index,method in enumerate(METHODS):
        o=(OUT/f"{method}.log").open("a",encoding="utf-8",buffering=1);e=(OUT/f"{method}.err.log").open("a",encoding="utf-8",buffering=1)
        command=[sys.executable,"-B","-u",str(ROOT/"experiments/train_gfs_rme128.py"),"train","--dataset","GFS128","--method",method]
        p=subprocess.Popen(command,cwd=ROOT,stdout=o,stderr=e)
        while p.poll() is None:
            status({"state":"training","active":method,"active_pid":p.pid,"completed":done,"queued":list(METHODS[index+1:]),"epochs_each":35,"updated_at":time.strftime("%Y-%m-%d %H:%M:%S"),"elapsed_seconds":time.time()-started});time.sleep(20)
        o.close();e.close();done[method]={"exit_code":p.returncode,"state":"complete" if p.returncode==0 else "failed"}
        if p.returncode:status({"state":"failed","active":None,"completed":done});return p.returncode
    code=subprocess.call([sys.executable,"-B",str(ROOT/"experiments/analyze_gfs128_wwfca_v42.py")],cwd=ROOT)
    status({"state":"complete","active":None,"completed":done,"epochs_each":35,"analysis_exit_code":code,"updated_at":time.strftime("%Y-%m-%d %H:%M:%S"),"elapsed_seconds":time.time()-started});return code
if __name__=="__main__":raise SystemExit(main())
