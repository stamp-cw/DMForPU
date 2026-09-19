"""Independent small-data HF diffusion study inspired by Chen et al. CVPR 2024."""
from pathlib import Path
import sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import argparse
import copy
import csv
import hashlib
import json
import math
import time
import subprocess
import numpy as np
import scipy.io as sio
import torch
from diffusion.chen_hf_diffusion import ChenHFConfig,ChenHFDiffusion,wrap,paired_recorruption,sr_loss,sd_loss
from experiments.evaluate_traditional_baselines import metrics

OUT=ROOT/'experiments/results/chen_hf_diffusion'
VARIANTS={
    'hf_base':dict(physics=False,sr=False,sd=False),
    'physics':dict(physics=True,sr=False,sd=False),
    'sr':dict(physics=False,sr=True,sd=False),
    'physics_sr':dict(physics=True,sr=True,sd=False),
    'full':dict(physics=True,sr=True,sd=True),
    'no_sparse':dict(physics=True,sparse=False,sr=True,sd=True),
    'no_cam':dict(physics=True,adaptive=False,sr=True,sd=True),
}


def dump(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8');tmp.replace(path)


def status(stage,**data):
    record={'stage':stage,'time':time.strftime('%Y-%m-%d %H:%M:%S'),**data}
    dump(OUT/'status.json',record);print(record,flush=True)


def manifest():
    path=OUT/'manifest.json'
    if not path.exists():
        source=json.loads((ROOT/'experiments/results/revision_3000/manifest.json').read_text(encoding='utf-8'))
        splits=source['domains']['synthetic']
        dump(path,{'domain':'synthetic','seed':42,'train':splits['train'][:300],'val':splits['val'][:60],
             'test':splits['test'][:60],'note':'420 fixed pairs at native 128x128; source official test never used for training'})
    return json.loads(path.read_text(encoding='utf-8'))


def load_data(split,device='cuda'):
    entries=manifest()[split];w=[];t=[]
    for e in entries:
        w.append(np.asarray(sio.loadmat(ROOT/e['input'])['input'],dtype=np.float32))
        t.append(np.asarray(sio.loadmat(ROOT/e['target'])[e['target_key']],dtype=np.float32))
    return tuple(torch.from_numpy(np.stack(x))[:,None].to(device) for x in (w,t))


def generator(seed,device='cuda'):return torch.Generator(device=device).manual_seed(seed)


def observe(w,snr,gen):
    sigma=torch.sqrt(10**.1/torch.pow(10.,snr/10))
    sigma=torch.where(snr>100,torch.zeros_like(sigma),sigma)
    noise=torch.randn(w.shape,device=w.device,generator=gen)
    noisy=wrap(w+sigma[:,None,None,None]*noise)
    return torch.where((snr>100)[:,None,None,None],w,noisy),sigma


def make_config(variant):
    return ChenHFConfig(**{k:v for k,v in VARIANTS[variant].items() if k in ('physics','sparse','adaptive')})


def train(variant,seed,epochs=20):
    folder=OUT/'runs'/f'{variant}_seed{seed}'
    if (folder/'complete.json').exists():return
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    cfg=make_config(variant);flags=VARIANTS[variant];net=ChenHFDiffusion(cfg).cuda()
    optimizer=torch.optim.AdamW(net.parameters(),lr=2e-4,weight_decay=0)
    scaler=torch.amp.GradScaler('cuda');teacher=None;history=[];start=0;best=math.inf
    phase=epochs*7//10;manifest();digest=hashlib.sha256((OUT/'manifest.json').read_bytes()).hexdigest()
    protocol={'variant':variant,'seed':seed,'epochs':epochs,'distill_start':phase,'batch':8,
              'lr':2e-4,'sr_weight':.05,'sd_weight':.05,'config':net.config_dict(),'manifest_sha256':digest}
    dump(folder/'protocol.json',protocol)
    if (folder/'last.pth').exists():
        state=torch.load(folder/'last.pth',map_location='cpu',weights_only=False)
        if state['protocol']!=protocol:raise ValueError('Resume protocol mismatch')
        net.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);scaler.load_state_dict(state['scaler'])
        history=state['history'];start=state['epoch']+1;best=state['best']
        if state['teacher'] is not None:
            teacher=copy.deepcopy(net).eval().requires_grad_(False);teacher.load_state_dict(state['teacher'])
        del state
    w,t=load_data('train');vw,vt=load_data('val');started=time.perf_counter()
    for epoch in range(start,epochs):
        if flags['sd'] and epoch>=phase and teacher is None:teacher=copy.deepcopy(net).eval().requires_grad_(False)
        gen=generator(seed*100000+epoch);order=torch.randperm(len(w),device='cuda',generator=gen)
        options=torch.tensor([1000.,1000.,1000.,1000.,1000.,30.,20.,10.,5.,0.],device='cuda')
        snrs=options[torch.randint(10,(len(w),),device='cuda',generator=gen)]
        observations,sigmas=observe(w,snrs,gen)
        # All stochastic training tensors drawn before variants take different forward paths.
        timesteps=torch.randint(cfg.train_steps,(len(w),),device='cuda',generator=gen)
        diffusion_noise=torch.randn(t.shape,device='cuda',generator=gen)
        rec_noise=torch.randn(w.shape,device='cuda',generator=gen)
        sums={'total':0.,'supervised':0.,'sr':0.,'sd':0.};updates=0;net.train();begin=time.perf_counter()
        for offset in range(0,len(w),8):
            idx=order[offset:offset+8];wi=observations[idx];target=t[idx];sigma=sigmas[idx];ts=timesteps[idx]
            x0=net.normalize(target);xt=net.scheduler.add_noise(x0,diffusion_noise[idx],ts)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16):
                estimate,phi,_=net(xt,wi,ts,sigma)
                supervised=(estimate-x0).square().mean();sr=supervised.new_zeros(());sd=supervised.new_zeros(())
                if flags['sr'] or teacher is not None:
                    plus,negative=paired_recorruption(wi,sigma,rec_noise[idx])
                    if flags['sr']:
                        _,_,stages=net(xt,plus,ts,sigma)
                        weights=[1/(len(stages)-j) for j in range(len(stages))]
                        sr=sum(sr_loss(p,negative)*weight for p,weight in zip(stages,weights))/sum(weights)
                    if teacher is not None:
                        with torch.no_grad():_,pseudo,_=teacher(xt,plus,ts,sigma)
                        sd=sd_loss(phi,pseudo)
                loss=supervised+.05*sr+.05*sd
            if not torch.isfinite(loss):raise RuntimeError(f'Nonfinite loss {variant} {epoch}')
            scaler.scale(loss).backward();scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(net.parameters(),1.)
            scale=scaler.get_scale();scaler.step(optimizer);scaler.update();updates+=int(scaler.get_scale()>=scale)
            for key,value in zip(sums,(loss,supervised,sr,sd)):sums[key]+=float(value.detach())*len(idx)
        net.eval();validation=[]
        # Fixed no-extra-noise and 10 dB validation; no test-selected hyperparameters.
        for snr in (1000.,10.):
            valid,sigma=observe(vw,torch.full((len(vw),),snr,device='cuda'),generator(900+int(snr)))
            gen_val=generator(8000+seed)
            for offset in range(0,len(vw),8):
                with torch.autocast('cuda',dtype=torch.float16):p=net.sample(valid[offset:offset+8],sigma[offset:offset+8],gen_val)
                e=p-vt[offset:offset+8];k=torch.round((-e/(2*math.pi)).flatten(1).median(1).values)[:,None,None,None]
                validation.extend((e+2*math.pi*k).abs().mean((1,2,3)).cpu().tolist())
        val=float(np.mean(validation));elapsed=time.perf_counter()-begin
        if not math.isfinite(val) or updates==0:raise RuntimeError('No successful updates or invalid validation')
        improved=val<best and (not flags['sd'] or epoch>=phase)
        if improved:best=val
        history.append({'epoch':epoch+1,**{k:v/len(w) for k,v in sums.items()},'val_mae':val,'seconds':elapsed,
                        'distilling':teacher is not None,'successful_updates':updates})
        state={'model':net.state_dict(),'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),
               'teacher':teacher.state_dict() if teacher is not None else None,'protocol':protocol,
               'history':history,'epoch':epoch,'best':best}
        temp=folder/'last.tmp.pth';torch.save(state,temp);temp.replace(folder/'last.pth')
        if improved:torch.save({'model':net.state_dict(),'protocol':protocol,'epoch':epoch},folder/'best.pth')
        dump(folder/'history.json',history);status('training',variant=variant,seed=seed,epoch=epoch+1,epochs=epochs,val_mae=val,seconds=elapsed)
    dump(folder/'complete.json',{'from_scratch':True,'epochs':epochs,'best_validation_mae':best,'seconds_this_session':time.perf_counter()-started})


def evaluate(variant,seed):
    output=OUT/'evaluation'/f'{variant}_seed{seed}.json'
    if output.exists():return
    state=torch.load(OUT/'runs'/f'{variant}_seed{seed}'/'best.pth',map_location='cpu',weights_only=False)
    chosen_epoch=state['epoch']+1
    net=ChenHFDiffusion(make_config(variant)).cuda().eval();net.load_state_dict(state['model']);del state
    w,t=load_data('test');rows=[];images={}
    for snr in (1000.,30.,20.,10.,5.,0.):
        observed,sigma=observe(w,torch.full((len(w),),snr,device='cuda'),generator(91000+int(snr)))
        for i in range(len(w)):
            with torch.autocast('cuda',dtype=torch.float16):p=net.sample(observed[i:i+1],sigma[i:i+1],generator(42000+i))
            pred=p[0,0].cpu().numpy();target=t[i,0].cpu().numpy();wi=observed[i,0].cpu().numpy()
            row={'sample':manifest()['test'][i]['id'],'snr':'clean' if snr>100 else str(int(snr)),**metrics(pred.astype(float),target.astype(float),wi.astype(float))}
            rows.append(row)
            if i==0:images.update({f'{row["snr"]}_prediction':pred,f'{row["snr"]}_target':target,f'{row["snr"]}_wrapped':wi})
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.with_suffix('.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary={}
    for snr in ('clean','30','20','10','5','0'):
        selected=[r for r in rows if r['snr']==snr]
        summary[snr]={k:float(np.mean([r[k] for r in selected])) for k in selected[0] if k not in ('sample','snr')}
        a=np.array([r['aligned_mae'] for r in selected]);rng=np.random.default_rng(42)
        summary[snr]['mae_ci95']=np.quantile(a[rng.integers(len(a),size=(1000,len(a)))].mean(1),[.025,.975]).tolist()
    np.savez_compressed(output.with_suffix('.npz'),**images)
    dump(output,{'variant':variant,'seed':seed,'samples':len(w),'summary':summary,'parameters':sum(p.numel() for p in net.parameters()),
                  'inference_steps':net.cfg.inference_steps,'selected_epoch':chosen_epoch,
                  'selection':'best validation; SD variants restricted to last 6 epochs',
                  'source':'fresh HF UNet2DModel without cross-attention, no FDU'})
    status('evaluated',variant=variant,seed=seed);report()


def report():
    results=[json.loads(p.read_text(encoding='utf-8')) for p in (OUT/'evaluation').glob('*.json')]
    lines=['# Chen-inspired HF diffusion 小批测试','',f'完成 {len(results)}/14 组评估。420 对 Synthetic 原始 128×128 数据：300/60/60；2 个种子；每组 20 轮，SD 最后 6 轮。',
           '直接使用无交叉注意力的 HF UNet2DModel + DDPMScheduler，不导入 FDU。监督 x₀ MSE 始终保留，不能称为无监督复现。',
           '所有组使用相同固定输入噪声、扩散噪声与采样种子；每次测试 5 步、单次采样；整数 2π 对齐仅用于评价。',
           '', '| 变体 | 完成种子数 | clean MAE | 10 dB MAE | 0 dB MAE | clean PGE |','|---|---:|---:|---:|---:|---:|']
    for variant in VARIANTS:
        selected=[r for r in results if r['variant']==variant]
        if selected:
            values=[np.mean([r['summary'][s][m] for r in selected]) for s,m in [('clean','aligned_mae'),('10','aligned_mae'),('0','aligned_mae'),('clean','pge')]]
            cells=[]
            for snr,key in [('clean','aligned_mae'),('10','aligned_mae'),('0','aligned_mae'),('clean','pge')]:
                a=[r['summary'][snr][key] for r in selected]
                cells.append(f'{np.mean(a):.4f} ± {np.std(a,ddof=1):.4f}' if len(a)>1 else f'{a[0]:.4f}')
            lines.append(f'| {variant} | {len(selected)} | '+' | '.join(cells)+' |')
    lines += ['','比较含义：hf_base→physics 测物理迭代；hf_base→sr 测配对重建；physics_sr→full 测蒸馏；full→no_sparse/no_cam 测异常值和噪声自适应。',
              '两种子均值仅作小规模方向筛选，逐图 CI 与各自结果保留于 evaluation。未测出提升时不宣称提升。真实域泛化和充分收敛未由本轮证明。',
              '来源与适配差异见 docs/CHEN_HF_DIFFUSION.zh-CN.md。本实验与 revision_3000 的数据量、骨干及预算不同，不能直接作公平排名。']
    lines += ['','表中 ± 为两个训练种子的样本标准差，不是置信区间。单种子逐图 bootstrap 区间见 JSON。']
    lines += ['CUDA 使用 FP16 与 CuDNN benchmark；固定了模型初始化和输入随机数，但不保证不同进程逐位一致。小幅消融差异还需更多种子或确定性复核。',
              'SD 组从最后 6 轮选模，其余组可从全部 20 轮选模；因此选择窗口不同也可能影响比较。']
    lines += ['','## 实际训练成本','','| 变体 | 参数量 | 两种子平均训练+验证秒数 |','|---|---:|---:|']
    for variant in VARIANTS:
        selected=[r for r in results if r['variant']==variant]
        seconds=[]
        for r in selected:
            h=json.loads((OUT/'runs'/f'{variant}_seed{r["seed"]}'/'history.json').read_text())
            seconds.append(sum(x['seconds'] for x in h))
        if selected:lines.append(f'| {variant} | {selected[0]["parameters"]} | {np.mean(seconds):.1f} |')
    lines += ['','成本不含首次 CUDA 初始化、数据读取、checkpoint 写盘和测试；SR/SD 组包含额外前向，非等计算量实验。']
    if len(results)==14:
        lines += ['','## 相对于 HF 基线的差值','','负值表示 MAE 降低；小批、两种子结果不能保证全量泛化。',
                  '', '| 变体 | clean ΔMAE | 10 dB ΔMAE | 0 dB ΔMAE |','|---|---:|---:|---:|']
        base=[r for r in results if r['variant']=='hf_base']
        for variant in VARIANTS:
            if variant=='hf_base':continue
            chosen=[r for r in results if r['variant']==variant]
            diffs=[np.mean([r['summary'][snr]['aligned_mae'] for r in chosen])-np.mean([r['summary'][snr]['aligned_mae'] for r in base]) for snr in ('clean','10','0')]
            lines.append('| '+variant+' | '+' | '.join(f'{x:+.4f}' for x in diffs)+' |')
        by_key={(r['variant'],r['seed']):r for r in results}
        lines += ['','## 本轮结论与稳定性','','- no_sparse（物理迭代 + CAM + SR + SD，不含稀疏 E）平均 MAE 最低，可作为下一轮候选；不能据此证明稀疏机制在其他数据上无效。',
                  '- 完整模型平均 PGE 更低，但 0 dB MAE 高于基线；SR/SD 没有呈现全面、稳定的收益。',
                  '- 种子差异很大。下面保留 no_sparse 与基线的逐种子结果，不能只看平均百分比。',
                  '', '| 种子 | 基线 clean MAE | no_sparse clean MAE | 差值（负值更好） |','|---|---:|---:|---:|']
        for seed in (42,43):
            b=by_key[('hf_base',seed)]['summary']['clean']['aligned_mae'];v=by_key[('no_sparse',seed)]['summary']['clean']['aligned_mae']
            lines.append(f'| {seed} | {b:.4f} | {v:.4f} | {v-b:+.4f} |')
        lines += ['','这轮只证明实现可训练并得到初步消融结果，尚未证明稳定提精度。进一步确认应增加训练种子和预算、从共同阶段快照分叉检验 SD，并保留新的独立测试集。']
        create_figures(results)
    (OUT/'REPORT.zh-CN.md').write_text('\n'.join(lines),encoding='utf-8')


def create_figures(results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figures=OUT/'figures';figures.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for variant in VARIANTS:
        chosen=[r for r in results if r['variant']==variant]
        snrs=('0','5','10','20','30','clean')
        axes[0].plot(range(6),[np.mean([r['summary'][s]['aligned_mae'] for r in chosen]) for s in snrs],'o-',label=variant)
        h=json.loads((OUT/'runs'/f'{variant}_seed42'/'history.json').read_text())
        axes[1].plot([r['epoch'] for r in h],[r['val_mae'] for r in h],label=variant)
    axes[0].set_xticks(range(6),('0','5','10','20','30','clean'));axes[0].set(xlabel='Added phase noise SNR (dB)',ylabel='Aligned MAE (rad)')
    axes[1].set(xlabel='Epoch (seed 42)',ylabel='Validation MAE (rad)');axes[1].axvline(14,color='gray',ls='--',label='Teacher snapshot')
    axes[0].legend(fontsize=8);axes[1].legend(fontsize=8);fig.tight_layout();fig.savefig(figures/'noise_and_validation.png',dpi=170);plt.close(fig)
    for snr in ('clean','5','0'):
        preds={}
        for variant in VARIANTS:
            with np.load(OUT/'evaluation'/f'{variant}_seed42.npz') as data:
                t=data[f'{snr}_target'];p=data[f'{snr}_prediction'];w=data[f'{snr}_wrapped']
                preds[variant]=p+np.rint(np.median(t-p)/(2*np.pi))*2*np.pi
        row=int(np.argmax(np.abs(np.diff(t,axis=1)).mean(1)))
        fig,ax=plt.subplots(figsize=(11,5));ax.plot(t[row],color='black',lw=2,label='reference')
        for name,p in preds.items():ax.plot(p[row],label=name,alpha=.8)
        ax.legend(ncol=4);ax.set(xlabel='Column',ylabel='Phase (rad)',title=f'Fixed test image; {snr}; reference-selected row {row}')
        fig.tight_layout();fig.savefig(figures/f'profile_{snr}.png',dpi=160);plt.close(fig)
        fig,axes=plt.subplots(2,4,figsize=(16,8));limit=max(np.quantile(np.abs(p-t),.99) for p in preds.values())
        for ax,(name,p) in zip(axes.flat,preds.items()):
            im=ax.imshow(np.abs(p-t),cmap='magma',vmin=0,vmax=limit);ax.set_title(name);ax.axis('off')
        axes.flat[-1].axis('off');fig.colorbar(im,ax=axes.ravel().tolist(),label='Aligned absolute error (rad)',shrink=.8)
        fig.savefig(figures/f'errors_{snr}.png',dpi=150,bbox_inches='tight');plt.close(fig)


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['train','evaluate','queue','report'])
    p.add_argument('--variant',choices=list(VARIANTS),default='full');p.add_argument('--seed',type=int,default=42)
    args=p.parse_args();torch.set_num_threads(4);torch.backends.cudnn.benchmark=True;OUT.mkdir(parents=True,exist_ok=True);manifest()
    if args.command=='train':train(args.variant,args.seed)
    elif args.command=='evaluate':evaluate(args.variant,args.seed)
    elif args.command=='report':report()
    else:
        for seed in (42,43):
            for variant in VARIANTS:
                for command in ('train','evaluate'):
                    with (OUT/'run.log').open('a',encoding='utf-8') as log:
                        result=subprocess.run([sys.executable,'-B','-u',__file__,command,'--variant',variant,'--seed',str(seed)],stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
                    if result.returncode:
                        status('failed',variant=variant,seed=seed,command=command,code=result.returncode)
                        raise RuntimeError('Study failed; see run.log')
        report();status('complete',trained_runs=14,evaluations=14)


if __name__=='__main__':main()
