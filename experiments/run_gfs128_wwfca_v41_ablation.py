"""Run WWFCA-v4.1 and its exact disabled control sequentially."""
from __future__ import annotations
import json, subprocess, sys, time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"experiments/results/gfs128_wwfca_v41"
METHODS=("wwfca_v41","wwfca_v41_off")

def write_status(value):
    OUT.mkdir(parents=True,exist_ok=True);tmp=OUT/"schedule_status.json.tmp"
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding="utf-8");tmp.replace(OUT/"schedule_status.json")

def main():
    OUT.mkdir(parents=True,exist_ok=True);started=time.time();results={}
    for index,method in enumerate(METHODS):
        stdout=(OUT/f"{method}.log").open("a",encoding="utf-8",buffering=1)
        stderr=(OUT/f"{method}.err.log").open("a",encoding="utf-8",buffering=1)
        command=[sys.executable,"-B","-u",str(ROOT/"experiments/train_gfs_rme128.py"),
                 "train","--dataset","GFS128","--method",method]
        process=subprocess.Popen(command,cwd=ROOT,stdout=stdout,stderr=stderr)
        while process.poll() is None:
            write_status({"state":"training","active":method,"active_pid":process.pid,
                          "completed":results,"queued":list(METHODS[index+1:]),
                          "updated_at":time.strftime("%Y-%m-%d %H:%M:%S"),
                          "elapsed_seconds":time.time()-started,"epochs_each":35})
            time.sleep(20)
        stdout.close();stderr.close();results[method]={"exit_code":process.returncode,
            "state":"complete" if process.returncode==0 else "failed"}
        if process.returncode:
            write_status({"state":"failed","active":None,"completed":results,
                          "updated_at":time.strftime("%Y-%m-%d %H:%M:%S"),
                          "elapsed_seconds":time.time()-started});return process.returncode
    analysis=subprocess.call([sys.executable,"-B",str(ROOT/"experiments/analyze_gfs128_wwfca_v41.py")],cwd=ROOT)
    write_status({"state":"complete","active":None,"completed":results,
                  "updated_at":time.strftime("%Y-%m-%d %H:%M:%S"),
                  "elapsed_seconds":time.time()-started,"epochs_each":35,"analysis_exit_code":analysis})
    return analysis

if __name__=="__main__":raise SystemExit(main())
