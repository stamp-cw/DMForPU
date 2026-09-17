"""Post-training measurements and reports for revision_study.py (no old weights)."""
from __future__ import annotations
import argparse
import csv
import gc
import json
import math
from pathlib import Path
import sys
import time
import warnings

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.revision_study import (ROOT, OUT, DEEP, METHODS, DOMAINS, prepare,
    read_pair, load_adapter, metric_row, seed_all, dump)

ANALYSIS = OUT/'analysis'


def release():
    gc.collect(); torch.cuda.empty_cache()


def save_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer=csv.DictWriter(f, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def infer(adapter, w, steps=5, snr=30):
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
        return adapter.predict(w, steps=steps, snr=snr)


def efficiency():
    path=ANALYSIS/'efficiency.json'
    if path.exists():return
    manifest=prepare();rows=[]
    for domain in DOMAINS:
        w,_=read_pair(manifest['domains'][domain]['test'][0])
        for method in (*DEEP,*METHODS):
            release()
            if method in METHODS:
                fn=METHODS[method];fn(w);times=[]
                for _ in range(20):
                    start=time.perf_counter();fn(w);times.append((time.perf_counter()-start)*1000)
                row={'domain':domain,'method':method,'device':'CPU','parameters':0,
                     'latency_ms_median':float(np.median(times)),'latency_ms_p95':float(np.quantile(times,.95)),
                     'peak_allocated_mb':None,'incremental_peak_mb':None,'counted_gflops':None,
                     'flops_note':'CPU algorithm; not counted','resolution':list(w.shape)}
            else:
                adapter=load_adapter(domain,method);tensor=torch.from_numpy(w)[None,None].cuda()
                for _ in range(5):infer(adapter,tensor)
                torch.cuda.synchronize();base=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
                times=[]
                for _ in range(20):
                    torch.cuda.synchronize();start=time.perf_counter();infer(adapter,tensor);torch.cuda.synchronize()
                    times.append((time.perf_counter()-start)*1000)
                peak=torch.cuda.max_memory_allocated();flops=None;note=''
                try:
                    from torch.utils.flop_counter import FlopCounterMode
                    with FlopCounterMode(display=False) as counter:infer(adapter,tensor)
                    flops=counter.get_total_flops()/1e9
                    note='Counted operators only; unsupported elementwise/FFT/control flow may be omitted. FDU includes 5 reverse steps.'
                except Exception as exc:note=f'Counter unavailable: {type(exc).__name__}: {exc}'
                row={'domain':domain,'method':method,'device':torch.cuda.get_device_name(),
                     'parameters':sum(p.numel() for p in adapter.model.parameters()),
                     'latency_ms_median':float(np.median(times)),'latency_ms_p95':float(np.quantile(times,.95)),
                     'peak_allocated_mb':peak/2**20,'incremental_peak_mb':(peak-base)/2**20,
                     'counted_gflops':flops,'flops_note':note,'resolution':list(w.shape)}
                del adapter,tensor
            rows.append(row);print('efficiency',domain,method,flush=True)
    dump(path,{'warmup_deep':5,'warmup_traditional':1,'repeats':20,'batch_size':1,'precision':'deep: fp16 autocast; traditional: numpy CPU',
               'timing_boundary':'preloaded input to output; excludes disk IO and host/device transfers; includes scheduler',
               'torch':torch.__version__,'rows':rows})


def steps():
    manifest=prepare()
    for domain in DOMAINS:
        path=ANALYSIS/f'steps_{domain}.json'
        if path.exists():continue
        release();adapter=load_adapter(domain,'fdu');rows=[]
        for count in (1,2,5,10,20):
            for i,entry in enumerate(manifest['domains'][domain]['test'][:100]):
                w,t=read_pair(entry);tensor=torch.from_numpy(w)[None,None].cuda();seed_all(42000+i)
                if i==0:
                    for _ in range(3):infer(adapter,tensor,count)
                    seed_all(42000+i)
                torch.cuda.synchronize();start=time.perf_counter();pred=infer(adapter,tensor,count);torch.cuda.synchronize()
                elapsed=(time.perf_counter()-start)*1000
                rows.append({'sample':entry['id'],'steps':count,'latency_ms':elapsed,
                             **metric_row(pred.squeeze().cpu().numpy(),t,w)})
            print('steps',domain,count,flush=True)
        save_rows(path.with_suffix('.csv'),rows)
        summary={str(s):{k:float(np.mean([r[k] for r in rows if r['steps']==s]))
                        for k in rows[0] if k not in ('sample','steps')} for s in (1,2,5,10,20)}
        dump(path,{'samples':100,'draws':1,'summary':summary});del adapter


def uncertainty():
    manifest=prepare()
    for domain in ('synthetic','insar','real'):
        path=ANALYSIS/f'uncertainty_{domain}.json'
        if path.exists():continue
        release();adapter=load_adapter('insar' if domain=='real' else domain,'fdu')
        entries=(manifest['real'] if domain=='real' else manifest['domains'][domain]['test'])[:100]
        rows=[];coverage=np.array([.1,.25,.5,.75,.9,1.])
        for i,entry in enumerate(entries):
            w,t=read_pair(entry);tensor=torch.from_numpy(w)[None,None].cuda();draws=[]
            torch.cuda.synchronize();start=time.perf_counter()
            for draw in range(20):
                seed_all(42000+i*100+draw);draws.append(infer(adapter,tensor).squeeze().cpu().numpy())
            torch.cuda.synchronize();elapsed=(time.perf_counter()-start)*1000
            samples=np.stack(draws).astype(np.float64);raw_std=samples.std(0,ddof=1)
            # Align draws to the first prediction, without using reference labels.
            shifts=np.rint(np.median((samples[0]-samples).reshape(20,-1),axis=1)/(2*np.pi))*2*np.pi
            samples+=shifts[:,None,None];mean=samples.mean(0);std=samples.std(0,ddof=1)
            k=np.rint(np.median(t-mean)/(2*np.pi));error=np.abs(mean+k*2*np.pi-t)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore');rho=float(spearmanr(std.ravel(),error.ravel()).statistic)
            order=np.argsort(std.ravel(),kind='stable');flat=error.ravel()
            row={'sample':entry['id'],'spearman':rho if np.isfinite(rho) else None,
                 'raw_std_mean':float(raw_std.mean()),'aligned_std_mean':float(std.mean()),
                 'ensemble_20_ms_including_host_copy':elapsed,
                 'single_mae':metric_row(draws[0],t,w)['aligned_mae'],
                 'ensemble_mae':float(error.mean())}
            row.update({f'risk_{c:g}':float(flat[order[:max(1,int(c*flat.size))]].mean()) for c in coverage})
            rows.append(row)
            if i==0:
                np.savez_compressed(ANALYSIS/f'uncertainty_{domain}_first.npz',wrapped=w,target=t,
                    mean=mean,std=std,raw_std=raw_std,absolute_error=error,draws=samples)
                fig,axes=plt.subplots(1,4,figsize=(16,4))
                for ax,(a,title) in zip(axes,[(t,'Reference'),(mean,'20-draw mean'),(std,'Aligned draw std'),(error,'Absolute error')]):
                    im=ax.imshow(a);ax.set_title(title);fig.colorbar(im,ax=ax);ax.axis('off')
                fig.tight_layout();fig.savefig(ANALYSIS/f'uncertainty_{domain}.png',dpi=150);plt.close(fig)
            if i%10==0:print('uncertainty',domain,i,flush=True)
        save_rows(path.with_suffix('.csv'),rows)
        values=[r['spearman'] for r in rows if r['spearman'] is not None]
        dump(path,{'samples':len(rows),'draws':20,'reverse_steps':5,'spearman_mean':float(np.mean(values)) if values else None,
                   'undefined_correlations':len(rows)-len(values),
                   'single_mae':float(np.mean([r['single_mae'] for r in rows])),
                   'ensemble_mae':float(np.mean([r['ensemble_mae'] for r in rows])),
                   'risk_coverage':{str(c):float(np.mean([r[f'risk_{c:g}'] for r in rows])) for c in coverage},
                   'interpretation':'Empirical draw variability, not calibrated posterior or confidence intervals; real reference uncertain.'})
        del adapter


def failures():
    """Select FDU's worst image by a declared metric; compare every method on it."""
    manifest=prepare();metadata=[]
    for domain in ('synthetic','insar','real'):
        entries=manifest['real'] if domain=='real' else manifest['domains'][domain]['test']
        with (OUT/'evaluation'/f'{domain}_fdu_both_seed42.csv').open(encoding='utf-8') as f:rows=list(csv.DictReader(f))
        for snr in (('clean',) if domain=='real' else ('5','0')):
            worst=max((r for r in rows if r['snr']==snr),key=lambda r:float(r['aligned_mae']))
            index=next(i for i,e in enumerate(entries) if e['id']==worst['sample'])
            w,t=read_pair(entries[index])
            if snr!='clean':
                rng=np.random.default_rng(42000+index+int(snr)*10000)
                w=np.angle(np.exp(1j*(w+rng.normal(0,math.sqrt(10**.1/10**(int(snr)/10)),w.shape)))).astype(np.float32)
            predictions={}
            for method in (*DEEP,*METHODS):
                if method in METHODS:p=METHODS[method](w)
                else:
                    release();adapter=load_adapter('insar' if domain=='real' else domain,method)
                    tensor=torch.from_numpy(w)[None,None].cuda();seed_all(42000+index)
                    p=infer(adapter,tensor,snr=30 if snr=='clean' else int(snr)).squeeze().cpu().numpy()
                    del adapter,tensor
                predictions[method]=p+np.rint(np.median(t-p)/(2*np.pi))*2*np.pi
            np.savez_compressed(ANALYSIS/f'failure_{domain}_{snr}.npz',wrapped=w,target=t,**predictions)
            row=int(np.argmax(np.abs(np.diff(t,axis=1)).mean(1)));fig,ax=plt.subplots(figsize=(12,5))
            ax.plot(t[row],label='reference',color='black',lw=2)
            for method,p in predictions.items():ax.plot(p[row],label=method,alpha=.8)
            ax.set(xlabel='Column',ylabel='Phase (rad)',title=f'{domain}; worst FDU aligned MAE; noise {snr}; row {row}')
            ax.legend(ncol=4);fig.tight_layout();fig.savefig(ANALYSIS/f'failure_{domain}_{snr}.png',dpi=160);plt.close(fig)
            metadata.append({'domain':domain,'snr':snr,'sample':entries[index]['id'],'reference_selected_row':row,
                             'selection':'highest FDU single-draw aligned MAE, seed 42; not selected independently per method'})
    dump(ANALYSIS/'failure_selection.json',metadata)


def figures():
    ANALYSIS.mkdir(parents=True,exist_ok=True)
    for domain in ('synthetic','insar','real'):
        path=ANALYSIS/f'uncertainty_{domain}.json'
        if path.exists():
            data=json.loads(path.read_text());curve=data['risk_coverage']
            fig,ax=plt.subplots(figsize=(6,4));ax.plot([float(c) for c in curve],list(curve.values()),'o-')
            ax.set(xlabel='Retained pixel fraction',ylabel='Mean absolute error (rad)',title=f'{domain}: uncertainty risk-coverage')
            fig.tight_layout();fig.savefig(ANALYSIS/f'risk_coverage_{domain}.png',dpi=160);plt.close(fig)
        path=ANALYSIS/f'steps_{domain}.json'
        if path.exists():
            data=json.loads(path.read_text())['summary'];fig,axes=plt.subplots(1,2,figsize=(10,4))
            for ax,key,label in zip(axes,('aligned_mae','latency_ms'),('MAE (rad)','Latency (ms)')):
                ax.plot([int(c) for c in data],[v[key] for v in data.values()],'o-');ax.set(xlabel='Reverse steps',ylabel=label)
            fig.suptitle(domain);fig.tight_layout();fig.savefig(ANALYSIS/f'steps_{domain}.png',dpi=160);plt.close(fig)
    for domain in ('synthetic','insar','real'):
        for snr in (('None',) if domain=='real' else ('None','5','0')):
            pairs=[]
            for method in (*DEEP,*METHODS):
                file=OUT/'evaluation'/f'{domain}_{method}_both_seed42_first.npz'
                if not file.exists():continue
                with np.load(file) as data:
                    target=data[f'{snr}_target'];pred=data[f'{snr}_prediction']
                    pred=pred+np.rint(np.median(target-pred)/(2*np.pi))*2*np.pi
                    pairs.append((method,pred))
            if not pairs:continue
            # Choose using only the reference, same row for all methods.
            row=int(np.argmax(np.abs(np.diff(target,axis=1)).mean(1)))
            fig,axes=plt.subplots(2,1,figsize=(12,8),sharex=True)
            axes[0].plot(target[row],color='black',lw=2,label='reference')
            for method,pred in pairs:
                axes[0].plot(pred[row],label=method,alpha=.8)
                axes[1].plot(pred[row]-target[row],label=method,alpha=.8)
            axes[0].legend(ncol=4);axes[0].set_ylabel('Phase (rad)');axes[1].set_ylabel('Error (rad)')
            axes[1].set_xlabel('Column');fig.suptitle(f'{domain}; added noise {snr}; reference-selected row {row}')
            fig.tight_layout();fig.savefig(ANALYSIS/f'profile_{domain}_{snr}.png',dpi=160);plt.close(fig)
            limit=max(float(np.quantile(np.abs(p-target),.99)) for _,p in pairs)
            fig,axes=plt.subplots(2,5,figsize=(20,8))
            for ax,(method,pred) in zip(axes.flat,pairs):
                im=ax.imshow(np.abs(pred-target),vmin=0,vmax=limit,cmap='magma');ax.set_title(method);ax.axis('off')
            for ax in axes.flat[len(pairs):]:ax.axis('off')
            fig.colorbar(im,ax=axes.ravel().tolist(),label='Absolute error (rad)',shrink=.8)
            fig.savefig(ANALYSIS/f'errors_{domain}_{snr}.png',dpi=140,bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(14,5))
    for path in sorted((OUT/'runs').glob('*/history.json')):
        history=json.loads(path.read_text());name=path.parent.name
        for ax,key in zip(axes,('loss','val_aligned_mae')):ax.plot([r['epoch'] for r in history],[r[key] for r in history],label=name)
    axes[0].set_title('Training objective (different scales)');axes[1].set_title('Validation aligned MAE')
    axes[1].legend(fontsize=5,ncol=2);fig.tight_layout();fig.savefig(ANALYSIS/'training_curves.png',dpi=180);plt.close(fig)


def report():
    files=sorted((OUT/'evaluation').glob('*.json'));results=[json.loads(f.read_text()) for f in files]
    trained=len(list((OUT/'runs').glob('*/complete.json')))
    lines=['# 3000 对数据重训练实验','',
      f'当前完成：{trained}/25 组训练、{len(results)}/41 组评估。以下仅列实际完成结果；未完成方法不能当作缺省零误差。', '',
      '固定划分：两个模拟域各 1200 训练 + 150 验证 + 150 测试；另测 100 对真实数据。所有深度权重独立初始化。',
      '这是有限预算试验（常规模型 30 轮，U3Net 50+20 轮），不能称为全量收敛复现；传统方法无需训练。',
      '下表误差按图平均；对齐只允许每图一个整数 2π，使用参考进行评价对齐。原始误差和实数偏移诊断保留在 CSV。',
      'clean 表示不额外注入噪声；InSAR 原始观测并非无噪声。真实参考与观测存在较大圆周差异，排名仅供诊断。','']
    for domain in ('synthetic','insar','real'):
        lines += [f'## {domain}：原始观测','', '| 方法 | 原始 MAE | 对齐 MAE [95% CI] | RMSE | NRMSE | PGE | 重缠绕 MAE |',
                  '|---|---:|---:|---:|---:|---:|---:|']
        selected=[r for r in results if r['domain']==domain and r['seed']==42 and r['variant']=='both']
        for r in sorted(selected,key=lambda r:r['summary']['clean']['aligned_mae']):
            s=r['summary']['clean'];lo,hi=s['aligned_mae_ci95']
            lines.append(f"| {r['method']} | {s['mae']:.4f} | {s['aligned_mae']:.4f} [{lo:.4f}, {hi:.4f}] | {s['aligned_rmse']:.4f} | {s['aligned_nrmse']:.4f} | {s['pge']:.4f} | {s['rewrap_circular_mae']:.4f} |")
        lines += ['','### 附加相位噪声：对齐 MAE','', '| 方法 | clean | 30 dB | 20 dB | 10 dB | 5 dB | 0 dB |','|---|---:|---:|---:|---:|---:|---:|'] if domain!='real' else []
        if domain!='real':
            for r in selected:lines.append('| '+r['method']+' | '+' | '.join(f"{r['summary'][s]['aligned_mae']:.4f}" for s in ('clean','30','20','10','5','0'))+' |')
        lines.append('')
    lines += ['## 三种子消融（Synthetic）','','| 变体 | 已完成种子数 | clean MAE 均值 ± 样本标准差 | 0 dB MAE |','|---|---:|---:|---:|']
    for variant in ('base','dcc','wwf','both'):
        chosen=[r for r in results if r['domain']=='synthetic' and r['method']=='fdu' and r['variant']==variant]
        if chosen:
            a=[r['summary']['clean']['aligned_mae'] for r in chosen];b=[r['summary']['0']['aligned_mae'] for r in chosen]
            sd=f'{np.std(a,ddof=1):.4f}' if len(a)>1 else 'NA'
            lines.append(f'| {variant} | {len(a)} | {np.mean(a):.4f} ± {sd} | {np.mean(b):.4f} |')
    efficiency_path=ANALYSIS/'efficiency.json'
    if efficiency_path.exists():
        efficiency_data=json.loads(efficiency_path.read_text())
        lines += ['','## 推理效率','','批量 1，深度方法 FP16，FDU 包含 5 步。CPU 传统方法与 GPU 方法分别标明，耗时不含磁盘和设备拷贝。',
          '', '| 域 | 方法 | 设备 | 参数量 | 中位延迟 ms | 峰值显存 MiB | 已计数 GFLOPs |','|---|---|---|---:|---:|---:|---:|']
        for r in efficiency_data['rows']:
            lines.append(f"| {r['domain']} | {r['method']} | {r['device']} | {r['parameters']} | {r['latency_ms_median']:.2f} | {r['peak_allocated_mb']} | {r['counted_gflops']} |")
    for domain in ('synthetic','insar','real'):
        upath=ANALYSIS/f'uncertainty_{domain}.json'
        if upath.exists():
            u=json.loads(upath.read_text())
            lines += ['',f"{domain} 不确定性：{u['samples']} 图 × {u['draws']} 次采样，平均 Spearman={u['spearman_mean']}；单次 MAE={u['single_mae']:.4f}，20 次均值 MAE={u['ensemble_mae']:.4f}。"]
    lines += ['','## 边界与后续','','- image bootstrap 区间不等同于训练随机性；消融另用 3 个训练种子。',
        '- 检查验证曲线后再决定全量长训预算；不要把固定轮数视作所有方法均已收敛。',
        '- 相干度分层与复数 speckle 需要真实相干图/复数观测；尚缺这些输入。',
        '- 完整论文主稿未提供，修订草稿见 docs/REVISION_DRAFT.zh-CN.md。',
        '- FLOPs 为支持算子的计数，不能冒充无遗漏的完整算术量；WWF 高分辨率对照结构尚需单独实现和训练。',
        '- analysis 目录提供效率、步数、不确定性及剖面图；各结果仅在对应文件实际生成后成立。','']
    (OUT/'REPORT.zh-CN.md').write_text('\n'.join(lines),encoding='utf-8')
    dump(ANALYSIS/'report_status.json',{'evaluation_files':len(results),'trained_runs':len(list((OUT/'runs').glob('*/complete.json'))),
         'expected_trained_runs':25,'expected_evaluations':41,'updated':time.strftime('%Y-%m-%d %H:%M:%S')})


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['all','figures','report','efficiency','steps','uncertainty','failures','watch']);args=p.parse_args()
    ANALYSIS.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4);torch.backends.cudnn.benchmark=True
    if args.command=='watch':
        # No GPU work until every queued training/evaluation has finished.
        while True:
            report()
            if len(list((OUT/'runs').glob('*/complete.json')))==25 and len(list((OUT/'evaluation').glob('*.json')))==41:
                break
            time.sleep(30)
        efficiency();steps();uncertainty();failures();figures();report()
        dump(ANALYSIS/'complete.json',{'time':time.strftime('%Y-%m-%d %H:%M:%S')})
    elif args.command=='all':
        efficiency();steps();uncertainty();failures();figures();report()
        dump(ANALYSIS/'complete.json',{'time':time.strftime('%Y-%m-%d %H:%M:%S')})
    else:globals()[args.command]()


if __name__=='__main__':main()
