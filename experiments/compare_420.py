"""Same 420-pair protocol: native FDU architecture, matched HF, DLPU and Chen HF."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
import argparse
import csv
import hashlib
import json
import math
import shutil
import subprocess
import time
from types import SimpleNamespace as NS
import numpy as np
import torch
from torch import nn
from diffusers import UNet2DModel,DDPMScheduler
from diffusion.chen_hf_diffusion import ChenHFConfig,ChenHFDiffusion
import experiments.chen_hf_study as core
from experiments.evaluate_traditional_baselines import metrics

SOURCE=ROOT/'experiments/results/chen_hf_diffusion'
OUT=ROOT/'experiments/results/comparison_420'
NEW=('hf_matched','dcc_only','wwfca_only','fdu','dlpu')
REUSED={'hf_small':'hf_base','chen_full':'full','chen_no_sparse':'no_sparse'}
ALL=('hf_small','hf_matched','dcc_only','wwfca_only','fdu','dlpu','chen_full','chen_no_sparse')


class ComparisonModel(nn.Module):
    def __init__(self,variant,widths=(128,128,128,128),layers=2,cross_dim=384):
        super().__init__();self.variant=variant;self.cfg=ChenHFConfig(physics=False)
        self.spec={'variant':variant,'widths':list(widths),'layers':layers,'cross_dim':cross_dim}
        self.dcc=variant in ('dcc_only','fdu')
        if variant=='dlpu':
            from model.unet.dlpu import DLPUNet
            self.backbone=DLPUNet(NS())
        elif variant in ('fdu','wwfca_only'):
            from model.fdunet.fdunet import FDUNet
            cfg=NS(model=NS(sample_size=128,in_channels=1,out_channels=1,layers_per_block=layers,
                block_out_channels=list(widths),cross_attention_dim=cross_dim),
                diffusion=NS(repeat_channels=1,conditioning_channels=2 if self.dcc else 1))
            self.backbone=FDUNet(cfg)
        else:
            self.backbone=UNet2DModel(sample_size=128,in_channels=3 if self.dcc else 2,
                out_channels=1,layers_per_block=layers,block_out_channels=tuple(widths),
                down_block_types=('DownBlock2D','DownBlock2D','DownBlock2D','DownBlock2D'),
                up_block_types=('UpBlock2D','UpBlock2D','UpBlock2D','UpBlock2D'),add_attention=False)
        self.scheduler=DDPMScheduler(num_train_timesteps=1000,prediction_type='sample',clip_sample=False)

    def normalize(self,phi):return phi/(14*math.pi)*2-1
    def denormalize(self,x):return (x+1)/2*(14*math.pi)
    def config_dict(self):return {'architecture':self.spec,'phase_range':[0,14*math.pi],'prediction_type':'sample','clip_sample':False}

    def forward(self,noisy,w,t,sigma):
        if self.variant=='dlpu':
            phi=self.backbone(w).float();return self.normalize(phi),phi,[phi]
        cond=torch.cat((w.sin(),w.cos()),1) if self.dcc else w/math.pi
        model_input=torch.cat((noisy,cond),1)
        if self.variant in ('fdu','wwfca_only'):
            hidden=torch.zeros(len(w),1,self.spec['cross_dim'],device=w.device,dtype=w.dtype)
            x0=self.backbone(model_input,t,encoder_hidden_states=hidden).sample.float()
        else:x0=self.backbone(model_input,t).sample.float()
        phi=self.denormalize(x0);return x0,phi,[phi]

    @torch.no_grad()
    def sample(self,w,sigma,generator=None,steps=5):
        if self.variant=='dlpu':return self.backbone(w).float()
        scheduler=DDPMScheduler.from_config(self.scheduler.config);scheduler.set_timesteps(steps,device=w.device)
        x=torch.randn(w.shape,device=w.device,dtype=w.dtype,generator=generator)
        for t in scheduler.timesteps:
            pred,_,_=self(x,w,t,sigma);x=scheduler.step(pred,t,x,generator=generator).prev_sample
        return self.denormalize(x)


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    src=SOURCE/'manifest.json';dst=OUT/'manifest.json'
    if not dst.exists():shutil.copyfile(src,dst)
    if src.read_bytes()!=dst.read_bytes():raise ValueError('Data manifest differs from the requested 420 pairs')
    core.OUT=OUT
    return hashlib.sha256(dst.read_bytes()).hexdigest()


def train(variant,seed):
    prepare()
    # Reuse the exact existing observation/timestep generator and epoch implementation.
    core.VARIANTS={variant:dict(physics=False,sr=False,sd=False)}
    core.make_config=lambda _:ChenHFConfig(physics=False)
    core.ChenHFDiffusion=lambda _:ComparisonModel(variant)
    core.train(variant,seed,epochs=20)


def checkpoint(variant,seed,selection):
    source_variant=REUSED.get(variant,variant)
    folder=(SOURCE if variant in REUSED else OUT)/'runs'/f'{source_variant}_seed{seed}'
    return folder/('last.pth' if selection=='final' else 'best.pth')


def load(variant,seed,selection):
    digest=prepare();path=checkpoint(variant,seed,selection)
    state=torch.load(path,map_location='cpu',weights_only=False);p=state['protocol']
    if (p['manifest_sha256']!=digest or p['epochs']!=20 or p['batch']!=8 or p['seed']!=seed or p['lr']!=2e-4):
        raise ValueError(f'Incompatible source protocol: {path}')
    if variant in REUSED:
        if p['variant']!=REUSED[variant]:raise ValueError('Wrong source variant')
        model=ChenHFDiffusion(ChenHFConfig(**p['config']))
    else:model=ComparisonModel(variant)
    model.load_state_dict(state['model'],strict=True);model=model.cuda().eval()
    return model,{'checkpoint':str(path.relative_to(ROOT)),'checkpoint_epoch':state['epoch']+1,
                  'manifest_sha256':digest,'reused_same_protocol':variant in REUSED,'training_protocol':p}


def evaluate(variant,seed,selection):
    output=OUT/'evaluation'/f'{variant}_seed{seed}_{selection}.json'
    if output.exists():return
    model,provenance=load(variant,seed,selection)
    w,t=core.load_data('test');entries=core.manifest()['test'];rows=[];images={}
    for snr in (1000.,30.,20.,10.,5.,0.):
        obs,sigma=core.observe(w,torch.full((len(w),),snr,device='cuda'),core.generator(91000+int(snr)))
        for i in range(len(w)):
            with torch.autocast('cuda',dtype=torch.float16):p=model.sample(obs[i:i+1],sigma[i:i+1],core.generator(42000+i))
            pred=p[0,0].cpu().numpy();gt=t[i,0].cpu().numpy();wi=obs[i,0].cpu().numpy()
            if not np.isfinite(pred).all():raise RuntimeError('Nonfinite prediction')
            label='clean' if snr>100 else str(int(snr))
            rows.append({'sample':entries[i]['id'],'snr':label,**metrics(pred.astype(float),gt.astype(float),wi.astype(float))})
            if i==0:images.update({f'{label}_prediction':pred,f'{label}_target':gt,f'{label}_wrapped':wi})
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.with_suffix('.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary={}
    for snr in ('clean','30','20','10','5','0'):
        chosen=[r for r in rows if r['snr']==snr]
        summary[snr]={k:float(np.mean([r[k] for r in chosen])) for k in chosen[0] if k not in ('sample','snr')}
        a=np.array([r['aligned_mae'] for r in chosen]);rng=np.random.default_rng(42)
        summary[snr]['mae_ci95']=np.quantile(a[rng.integers(len(a),size=(1000,len(a)))].mean(1),[.025,.975]).tolist()
    np.savez_compressed(output.with_suffix('.npz'),**images)
    core.dump(output,{'variant':variant,'seed':seed,'selection':selection,'samples':60,'summary':summary,
        'parameters':sum(p.numel() for p in model.parameters()),'inference_steps':1 if variant=='dlpu' else 5,**provenance})
    core.status('evaluated',variant=variant,seed=seed,selection=selection);report()


def report():
    prepare();results=[json.loads(p.read_text(encoding='utf-8')) for p in (OUT/'evaluation').glob('*.json')]
    lines=['# 同一批 420 对数据的方法与组件对比','',f'已生成 {len(results)}/32 组评估（8 方法 × 2 种子 × final/best）。',
        '固定 300/60/60、128×128、20 轮、batch 8、AdamW lr=0.0002、FP16。主表统一第 20 轮；附表使用验证最佳。',
        '所有方法使用相同固定观测与指标；扩散模型固定 1000 时间步、sample prediction、5 步推理、单次采样、不裁剪。',
        '新训练的 FDU/DLPU 使用归一化相位 MSE 的共同筛选协议；这不是各论文原始 L1 训练配方或已充分调参的性能上限。',
        'hf_matched、dcc_only、wwfca_only、fdu 使用四层 128 通道同规格骨干；hf_small 和 Chen 改进使用此前三层 32/64/64 小骨干。跨两种骨干不能直接归因于模块。',
        'hf_small、chen_full、chen_no_sparse 复用同一批数据和协议下的既有权重，重新推理；其余五组×两种子全部新训。无测试集训练、无旧 3000 对大样本权重。',
        '均为逐图整数 2π 对齐误差；原始误差与图级 CI 见 JSON/CSV。± 表示两个训练种子的样本标准差。','']
    for selection in ('final','best'):
        lines += [f'## {selection}：'+('统一第 20 轮主表' if selection=='final' else '验证最佳参考表'),'',
                  '| 方法 | 种子数 | 参数量 | clean MAE | 10 dB MAE | 0 dB MAE | clean PGE |','|---|---:|---:|---:|---:|---:|---:|']
        for variant in ALL:
            chosen=[r for r in results if r['variant']==variant and r['selection']==selection]
            if not chosen:continue
            cells=[]
            for snr,key in [('clean','aligned_mae'),('10','aligned_mae'),('0','aligned_mae'),('clean','pge')]:
                a=[r['summary'][snr][key] for r in chosen]
                cells.append(f'{np.mean(a):.4f} ± {np.std(a,ddof=1):.4f}' if len(a)>1 else f'{a[0]:.4f}')
            lines.append(f'| {variant} | {len(chosen)} | {chosen[0]["parameters"]} | '+' | '.join(cells)+' |')
        lines.append('')
    lines += ['## 消融映射','','| DCC | WWFCA | 对应实验 |','|---|---|---|','| 无 | 无 | hf_matched |','| 有 | 无 | dcc_only |','| 无 | 有 | wwfca_only |','| 有 | 有 | fdu |','',
        'Chen full 含物理迭代、CAM、稀疏 E、SR、SD；chen_no_sparse 移除 E。Chen 变体保留额外 SR/SD 项并获得合成噪声 σ，其余模型未使用该噪声条件；不是等计算量/等先验信息比较。',
        '当前骨干、轮数和数据规模只能说明小批筛选表现。FP16/CuDNN 不保证跨进程逐位确定；两个种子仍不足以证明普适优势。',
        'best 中 SD 模型只从最后 6 轮选模，其余可从全部 20 轮选模；因此主结论以统一 final 为准。']
    final=[r for r in results if r['selection']=='final']
    if len(final)==16:
        indexed={(r['variant'],r['seed']):r for r in final}
        effects=[]
        lines += ['','## 同规格 2×2 消融的逐种子差值','','MAE 差值，负值为改善。不能把跨骨干的差异当作 DCC/WWFCA 效果。',
                  '', '| 种子 | 噪声 | DCC（无 WWFCA） | WWFCA（无 DCC） | DCC（有 WWFCA） | WWFCA（有 DCC） | 交互项 |',
                  '|---|---|---:|---:|---:|---:|---:|']
        for seed in (42,43):
            for snr in ('clean','10','0'):
                a,b,c,d=[indexed[(v,seed)]['summary'][snr]['aligned_mae'] for v in ('hf_matched','dcc_only','wwfca_only','fdu')]
                values=[b-a,c-a,d-c,d-b,d-b-c+a]
                lines.append(f'| {seed} | {snr} | '+' | '.join(f'{v:+.4f}' for v in values)+' |')
                effects.append({'seed':seed,'snr':snr,'dcc_without_wwfca':b-a,'wwfca_without_dcc':c-a,
                                'dcc_with_wwfca':d-c,'wwfca_with_dcc':d-b,'interaction':d-b-c+a})
        core.dump(OUT/'ablation_effects.json',effects)
        lines += ['','## 结果解读','','- DLPU 的 final MAE 最低且随附加噪声变化最小；其 clean PGE 在主表中最高，说明低绝对误差没有同时转化为最佳局部梯度。',
                  '- DCC-only 的 clean/10 dB MAE 和 PGE 优于同规格 HF，但两个种子的 0 dB MAE 都变差；DCC 的优势在强噪声下没有保持。',
                  '- WWFCA-only 的种子方差很大：seed42 明显退化，seed43 改善。完整 FDU 比同规格 HF 的两种子平均 clean MAE 更低，但未优于 DCC-only。',
                  '- Chen no_sparse 与 FDU 的平均 clean MAE 接近，在 0 dB 上更低；它使用小骨干、噪声条件和额外 SR/SD，不能将差值归因于单一结构。Chen full 没有超过 no_sparse。',
                  '- 剖面图显示各方法仍存在局部形状偏差；DLPU 的低 MAE 与较高 PGE 应同时报告，不能只用单一指标下结论。']
        figures(final)
    (OUT/'REPORT.zh-CN.md').write_text('\n'.join(lines),encoding='utf-8')


def figures(results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=OUT/'figures';folder.mkdir(exist_ok=True)
    snrs=('0','5','10','20','30','clean')
    fig,axes=plt.subplots(1,3,figsize=(18,5))
    groups=[('Matched FDU ablation',('hf_matched','dcc_only','wwfca_only','fdu')),
            ('Small HF and Chen improvements',('hf_small','chen_full','chen_no_sparse')),
            ('Method comparison',('hf_small','hf_matched','fdu','dlpu','chen_full','chen_no_sparse'))]
    for ax,(title,variants) in zip(axes,groups):
        for variant in variants:
            chosen=[r for r in results if r['variant']==variant]
            ax.plot(range(6),[np.mean([r['summary'][s]['aligned_mae'] for r in chosen]) for s in snrs],'o-',label=variant)
        ax.set_xticks(range(6),snrs);ax.set(title=title,xlabel='Added noise SNR (dB)',ylabel='Aligned MAE (rad)');ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(folder/'final_noise_comparison.png',dpi=160);plt.close(fig)
    for snr in ('clean','5','0'):
        preds={}
        for variant in ALL:
            with np.load(OUT/'evaluation'/f'{variant}_seed42_final.npz') as data:
                t=data[f'{snr}_target'];p=data[f'{snr}_prediction']
                preds[variant]=p+np.rint(np.median(t-p)/(2*np.pi))*2*np.pi
        row=int(np.argmax(np.abs(np.diff(t,axis=1)).mean(1)))
        fig,axes=plt.subplots(2,1,figsize=(12,8),sharex=True);axes[0].plot(t[row],color='black',lw=2,label='reference')
        for name,p in preds.items():
            axes[0].plot(p[row],label=name,alpha=.8);axes[1].plot(p[row]-t[row],label=name,alpha=.8)
        axes[0].legend(ncol=3,fontsize=8);axes[0].set(ylabel='Phase (rad)',title=f'Fixed first test image; seed42 final; {snr}; row {row}')
        axes[1].set(xlabel='Column',ylabel='Error (rad)');fig.tight_layout();fig.savefig(folder/f'profile_{snr}.png',dpi=160);plt.close(fig)
        fig,axes=plt.subplots(2,4,figsize=(16,8));limit=max(np.quantile(np.abs(p-t),.99) for p in preds.values())
        for ax,(name,p) in zip(axes.flat,preds.items()):
            im=ax.imshow(np.abs(p-t),vmin=0,vmax=limit,cmap='magma');ax.set_title(name);ax.axis('off')
        fig.colorbar(im,ax=axes.ravel().tolist(),shrink=.8,label='Aligned absolute error (rad)')
        fig.savefig(folder/f'errors_{snr}.png',dpi=150,bbox_inches='tight');plt.close(fig)


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['train','evaluate','queue','report'])
    p.add_argument('--variant',choices=ALL,default='fdu');p.add_argument('--seed',type=int,default=42)
    p.add_argument('--selection',choices=['final','best'],default='final');args=p.parse_args()
    prepare();torch.set_num_threads(4);torch.backends.cudnn.benchmark=True
    if args.command=='train':train(args.variant,args.seed)
    elif args.command=='evaluate':evaluate(args.variant,args.seed,args.selection)
    elif args.command=='report':report()
    else:
        jobs=[]
        for seed in (42,43):
            for variant in ALL:
                if variant in NEW:jobs.append(['train','--variant',variant,'--seed',str(seed)])
                for selection in ('final','best'):jobs.append(['evaluate','--variant',variant,'--seed',str(seed),'--selection',selection])
        core.dump(OUT/'queue.json',jobs)
        for index,job in enumerate(jobs):
            core.status('job_started',index=index,total=len(jobs),command=job)
            with (OUT/'run.log').open('a',encoding='utf-8') as log:
                r=subprocess.run([sys.executable,'-B','-u',__file__,*job],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            if r.returncode:
                core.status('failed',index=index,command=job,exit_code=r.returncode);raise RuntimeError('See run.log')
        report();core.status('complete',new_training_runs=10,reused_training_runs=6,evaluations=32)


if __name__=='__main__':main()
