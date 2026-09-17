"""Run the new authorized study, then resume the pre-existing comparison queue."""
from pathlib import Path
import subprocess
import sys
import json
import time
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/results/chen_hf_diffusion'
OUT.mkdir(parents=True,exist_ok=True)
try:
    with (OUT/'queue.log').open('a',encoding='utf-8') as log:
        code=subprocess.call([sys.executable,'-B','-u',str(ROOT/'experiments/chen_hf_study.py'),'queue'],
                             cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
finally:
    old=ROOT/'experiments/results/revision_3000'
    with (old/'queue_after_chen.log').open('a',encoding='utf-8') as log:
        proc=subprocess.Popen([sys.executable,'-B','-u',str(ROOT/'experiments/revision_study.py'),'queue'],
                              cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
    (OUT/'old_queue_resumed.json').write_text(json.dumps({'time':time.strftime('%Y-%m-%d %H:%M:%S'),'pid':proc.pid}),encoding='utf-8')
sys.exit(code)
