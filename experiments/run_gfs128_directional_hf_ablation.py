"""Train the directional high-frequency residual without cross-attention."""
from __future__ import annotations
import json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"experiments/results/gfs128_directional_hf_ablation";METHOD="wwfca_v41_noattn"
def status(value):
    OUT.mkdir(parents=True,exist_ok=True);target=OUT/"schedule_status.json";tmp=OUT/f"status.{os.getpid()}.tmp";tmp.write_text(json.dumps(value,indent=2),encoding="utf-8")
    for attempt in range(10):
        try:tmp.replace(target);return
        except PermissionError:
            if attempt==9:raise
            time.sleep(.1*(attempt+1))
def main():
    OUT.mkdir(parents=True,exist_ok=True);started=time.time();o=(OUT/f"{METHOD}.log").open("a",encoding="utf-8",buffering=1);e=(OUT/f"{METHOD}.err.log").open("a",encoding="utf-8",buffering=1)
    p=subprocess.Popen([sys.executable,"-B","-u",str(ROOT/"experiments/train_gfs_rme128.py"),"train","--dataset","GFS128","--method",METHOD],cwd=ROOT,stdout=o,stderr=e)
    while p.poll() is None:status({"state":"training","active":METHOD,"pid":p.pid,"epochs":35,"updated_at":time.strftime("%Y-%m-%d %H:%M:%S"),"elapsed_seconds":time.time()-started});time.sleep(20)
    o.close();e.close()
    if p.returncode:status({"state":"failed","exit_code":p.returncode});return p.returncode
    code=subprocess.call([sys.executable,"-B",str(ROOT/"experiments/analyze_gfs128_directional_hf_ablation.py")],cwd=ROOT)
    status({"state":"complete","active":None,"exit_code":0,"analysis_exit_code":code,"updated_at":time.strftime("%Y-%m-%d %H:%M:%S"),"elapsed_seconds":time.time()-started});return code
if __name__=="__main__":raise SystemExit(main())
