"""Summarize the completed official-source GFS128 rerun."""
from pathlib import Path
import csv,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'experiments/results/gfs128_upstream';RUNS=BASE/'runs'
METHODS=('u3net','dlpu','punet','uformer','restormer','vurnet','sqd_lstm');LABEL={'u3net':'U3Net','dlpu':'DLPU','punet':'PUNet','uformer':'Uformer-B','restormer':'Restormer','vurnet':'VUR-Net','sqd_lstm':'SQD-LSTM'};SNRS=(0,5,10,20,30)
rows=[]
for method in METHODS:
    result=json.loads((RUNS/method/'complete.json').read_text(encoding='utf-8'))
    for snr in SNRS:
        m=result['tests'][str(snr)];rows.append({'method':LABEL[method],'snr_db':snr,'mean_aligned_mae':m['mean_aligned_mae'],'mean_aligned_nrmse':m['mean_aligned_nrmse'],'u3_aligned_mae':m['u3_aligned_mae'],'u3_aligned_rmse':m['u3_aligned_rmse'],'u3_aligned_nrmse':m['u3_aligned_nrmse'],'u3_aligned_ssim':m['u3_aligned_ssim'],'raw_au':m['raw_au'],'integer_aligned_au':m['integer_aligned_au'],'range_aligned_au':m['range_aligned_au'],'raw_mae':m['raw_mae'],'rewrap_circular_mae':m['rewrap_circular_mae'],'parameters':result['parameters'],'official_commit':result['official_commit']})
with (BASE/'summary.csv').open('w',newline='',encoding='utf-8-sig') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(BASE/'summary.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
fig,axes=plt.subplots(1,2,figsize=(12,4.7))
for method in METHODS:
    r=[x for x in rows if x['method']==LABEL[method]];axes[0].plot(SNRS,[x['mean_aligned_mae'] for x in r],marker='o',label=LABEL[method]);axes[1].plot(SNRS,[x['mean_aligned_nrmse'] for x in r],marker='o',label=LABEL[method])
axes[0].set_ylabel('Mean-aligned MAE (rad)');axes[1].set_ylabel('Mean-aligned NRMSE');
for ax in axes:ax.set_xlabel('SNR (dB)');ax.grid(alpha=.25);ax.legend(frameon=False,fontsize=8)
fig.suptitle('GFS128 official-repository rerun');fig.tight_layout();fig.savefig(BASE/'comparison.png',dpi=180);plt.close(fig)
print(BASE/'summary.csv');print(BASE/'comparison.png')
