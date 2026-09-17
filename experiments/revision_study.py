"""Reproducible 3000-pair revision study. All checkpoints belong to this study."""
from __future__ import annotations
import argparse
import contextlib
import hashlib
import importlib
import json
import logging
import math
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import scipy.io as sio
import torch
import torch.nn.functional as F
import yaml
from diffusers import DDPMScheduler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
from utils.util import dict2namespace
from experiments.evaluate_traditional_baselines import metrics
from traditional.phase_unwrapping import METHODS

OUT = ROOT/'experiments/results/revision_3000'
DOMAINS = {'synthetic': ('SyntheticPUMat128Big', 'train_in', 'train_gt', 'test_in', 'test_gt', 'gt'),
           'insar': ('InSARDLPUMat256Big', 'train_wrapped', 'train_absolute', 'test_wrapped', 'test_absolute', 'output')}
DEEP = ('fdu', 'dlpu', 'punet', 'sqd_lstm', 'uformer', 'restormer', 'u3net')
SNRS = (None, 30, 20, 10, 5, 0)


def dump(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def status(stage, **values):
    dump(OUT/'status.json', {'stage': stage, 'time': time.strftime('%Y-%m-%d %H:%M:%S'), **values})
    print(stage, values, flush=True)


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare():
    if (OUT/'manifest.json').exists():
        return json.loads((OUT/'manifest.json').read_text(encoding='utf-8'))
    rng = np.random.default_rng(42)
    manifest = {'seed': 42, 'total_simulated_pairs': 3000, 'domains': {},
                'split_policy': '1200 train + 150 validation from official train; 150 from official test per domain'}
    audits = []
    for domain, (name, ti, tg, vi, vg, key) in DOMAINS.items():
        root = ROOT/'data'/name
        def pairs(inp, target):
            keys = sorted({p.stem for p in (root/inp).glob('*.mat')} & {p.stem for p in (root/target).glob('*.mat')})
            return [{'id': k, 'input': str((root/inp/f'{k}.mat').relative_to(ROOT)),
                     'target': str((root/target/f'{k}.mat').relative_to(ROOT)), 'target_key': key} for k in keys]
        train = pairs(ti,tg); test = pairs(vi,vg)
        if len(train)<1350 or len(test)<150: raise ValueError(f'Not enough pairs in {domain}')
        idx = rng.permutation(len(train))[:1350]; tidx = rng.permutation(len(test))[:150]
        manifest['domains'][domain] = {'train': [train[i] for i in idx[:1200]],
            'val': [train[i] for i in idx[1200:]], 'test': [test[i] for i in tidx]}
        if domain == 'insar': manifest['real'] = pairs('test_wrapped_real','test_absolute_real')
    for domain, splits in manifest['domains'].items():
        for split, entries in splits.items():
            for entry in entries:
                audits.append(audit_pair(domain, split, entry))
    for entry in manifest['real']: audits.append(audit_pair('real','test',entry))
    import csv
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT/'data_audit.csv').open('w', newline='', encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(audits[0]));writer.writeheader();writer.writerows(audits)
    summary = {}
    for domain in ('synthetic','insar','real'):
        rows=[r for r in audits if r['domain']==domain]
        summary[domain]={'pairs':len(rows), 'reference_cycle_mae_mean':float(np.mean([r['cycle_mae'] for r in rows])),
            'reference_cycle_mae_max':float(max(r['cycle_mae'] for r in rows)),
            'offset_corrected_cycle_mae_mean':float(np.mean([r['cycle_offset_mae'] for r in rows])),
            'nonfinite_pairs':sum(not r['finite'] for r in rows),
            'absolute_range':[min(r['target_min'] for r in rows),max(r['target_max'] for r in rows)]}
    dump(OUT/'data_audit.json',summary)
    dump(OUT/'manifest.json',manifest)
    status('data_prepared', summary=summary)
    return manifest


def read_pair(entry):
    w=np.asarray(sio.loadmat(ROOT/entry['input'])['input'],dtype=np.float32)
    t=np.asarray(sio.loadmat(ROOT/entry['target'])[entry['target_key']],dtype=np.float32)
    if w.ndim!=2 or w.shape!=t.shape or not (np.isfinite(w).all() and np.isfinite(t).all()):
        raise ValueError(f'Invalid data pair: {entry}')
    return w,t


def audit_pair(domain, split, entry):
    w,t=read_pair(entry)
    residual=np.angle(np.exp(1j*(t.astype(np.float64)-w)))
    offset=float(np.angle(np.exp(1j*residual).mean()))
    return {'domain':domain,'split':split,'id':entry['id'],'height':w.shape[0],'width':w.shape[1],
            'finite':True,'wrapped_min':float(w.min()),'wrapped_max':float(w.max()),
            'target_min':float(t.min()),'target_max':float(t.max()),
            'cycle_mae':float(np.abs(residual).mean()),'circular_offset':offset,
            'cycle_offset_mae':float(np.abs(np.angle(np.exp(1j*(residual-offset)))).mean()),
            'negated_target_cycle_mae':float(np.abs(np.angle(np.exp(1j*(-t.astype(np.float64)-w)))).mean()),
            'input_sha256':file_hash(ROOT/entry['input']),'target_sha256':file_hash(ROOT/entry['target'])}


def arrays(entries, device='cuda'):
    pairs=[read_pair(e) for e in entries]
    return tuple(torch.from_numpy(np.stack([p[i] for p in pairs]))[:,None].to(device) for i in (0,1))


def config_for(domain, method, epochs, seed, variant='both'):
    tag='synpu_128_big' if domain=='synthetic' else 'dlpu_256_big'
    cfg=yaml.safe_load((ROOT/f'configs/{method}_{tag}.yaml').read_text(encoding='utf-8'))
    cfg['training'].update(device='cuda',amp=True,brand_new_epochs=epochs,
        distill_start_epoch=max(1,epochs*5//7),distill_epochs=epochs-max(1,epochs*5//7))
    cfg['sampling']['device']='cuda'
    cfg['optim']['warmup']=0
    for key in ('lr','eps','weight_decay','beta1'):
        cfg['optim'][key]=float(cfg['optim'][key])
    cfg['seed']=seed
    if method=='fdu':
        cfg['diffusion']['prediction_type']='sample'
        cfg['diffusion']['conditioning_channels']=2 if variant in ('both','dcc') else 1
        cfg['model']['name']='FDUNet' if variant in ('both','wwf') else 'FDUNetV1'
    return cfg


class Adapter:
    def __init__(self,cfg,method,variant='both'):
        self.method=method;self.variant=variant;self.raw_config=cfg
        self.cfg=dict2namespace(cfg);self.cfg.logger=logging.getLogger('revision');self.cfg.mode='train_model'
        self.cfg.writer=None
        self.cfg.io=dict2namespace({'use_tensorboard':False,'use_wandb':False})
        if method=='fdu':
            from model.fdunet.fdunet import FDUNet
            from model.fdunet.fdunet_v1 import FDUNetV1
            self.model=(FDUNet if variant in ('both','wwf') else FDUNetV1)(self.cfg).cuda()
            self.scheduler=DDPMScheduler(num_train_timesteps=1000,prediction_type='sample')
        else:
            importlib.import_module({'dlpu':'model.unet.dlpu','punet':'model.unet.punet',
                'sqd_lstm':'model.lstm.sqd_lstm','uformer':'model.transformer.uformer',
                'restormer':'model.transformer.restormer','u3net':'model.u3net.u3net'}[method])
            from selector.model_selector import _MODELS
            self.model=_MODELS[self.cfg.model.name](self.cfg).cuda()
        self.teacher=None

    def cond(self,w):
        return torch.cat((w.sin(),w.cos()),1) if self.variant in ('both','dcc') else w/torch.pi

    def denoise(self,x,w,t):
        enc=torch.zeros(len(w),1,self.cfg.model.cross_attention_dim,device=w.device,dtype=w.dtype)
        return self.model(torch.cat((x,self.cond(w)),1),t,encoder_hidden_states=enc).sample

    @staticmethod
    def gradient(x):
        from model.u3net.u3net import grad_op
        return grad_op(x)

    def u3(self,w,snr,teacher=False):
        grad=self.gradient(w);grad=torch.atan2(grad.sin(),grad.cos())
        std=(10**.1/torch.pow(10.,snr.float()/10)).sqrt().reshape(-1,1)
        x=torch.ones_like(w);a=torch.zeros(*w.shape,2,device=w.device)
        return (self.teacher if teacher else self.model)(grad,std,x,a)

    def loss(self,w,target,snr,epoch):
        if self.method=='fdu':
            low=self.cfg.data.k_min*2*math.pi;scale=(self.cfg.data.k_max-self.cfg.data.k_min)*2*math.pi
            clean=((target-low)/scale).clamp(0,1)*2-1
            t=torch.randint(0,1000,(len(w),),device=w.device)
            noisy=self.scheduler.add_noise(clean,torch.randn_like(clean),t)
            return F.l1_loss(self.denoise(noisy,w,t).float(),clean)
        if self.method=='u3net':
            std=(10**.1/torch.pow(10.,snr.float()/10)).sqrt()[:,None,None,None]
            plus=w+torch.randn_like(w)*std;plus=torch.atan2(plus.sin(),plus.cos())
            if self.teacher is not None:
                with torch.no_grad(): teacher,_=self.u3(plus,snr,True)
                _,preds=self.u3(w,snr)
                return sum(F.l1_loss(self.gradient(p.float()),self.gradient(teacher.float()))/(len(preds)-i)
                           for i,p in enumerate(preds))
            _,preds=self.u3(plus,snr)
            negative=self.gradient(2*w-plus)
            return sum(torch.atan2((negative-self.gradient(p.float())).sin(),
                (negative-self.gradient(p.float())).cos()).square().mean()/(len(preds)-i) for i,p in enumerate(preds))
        p=self.model(w).float()
        if self.method=='sqd_lstm':
            e=p-target
            variance=e.square().mean((1,2,3))-e.mean((1,2,3)).square()
            tv=(e[:,:,1:]-e[:,:,:-1]).abs().mean()+(e[:,:,:,1:]-e[:,:,:,:-1]).abs().mean()
            return variance.mean()+.1*tv
        if self.method=='uformer': return ((p-target).square()+1e-6).sqrt().mean()
        return F.l1_loss(p,target)

    @torch.no_grad()
    def predict(self,w,snr=30,steps=5):
        if self.method=='fdu':
            self.scheduler.set_timesteps(steps,device=w.device)
            x=torch.randn_like(w)
            for t in self.scheduler.timesteps:
                pred=self.denoise(x,w,t).float()
                x=self.scheduler.step(pred,t,x).prev_sample
            return (x+1)/2*((self.cfg.data.k_max-self.cfg.data.k_min)*2*math.pi)+self.cfg.data.k_min*2*math.pi
        if self.method=='u3net':
            s=torch.full((len(w),),float(snr),device=w.device) if not torch.is_tensor(snr) else snr
            return self.u3(w,s)[0].float()
        return self.model(w).float()


def observed(w,snr,generator):
    noise=torch.randn(w.shape,device=w.device,generator=generator)
    std=(10**.1/torch.pow(10.,snr.float()/10)).sqrt()[:,None,None,None]
    x=w+noise*std
    return torch.where((snr>100)[:,None,None,None], w, torch.atan2(x.sin(),x.cos()))


def job_dir(domain,method,seed=42,variant='both'):
    return OUT/'runs'/f'{domain}_{method}_{variant}_seed{seed}'


def train(domain,method,epochs=30,seed=42,variant='both',batch_size=None):
    manifest=prepare();folder=job_dir(domain,method,seed,variant)
    if (folder/'complete.json').exists(): return
    folder.mkdir(parents=True,exist_ok=True)
    cfg=config_for(domain,method,epochs,seed,variant)
    dump(folder/'config.json',cfg)
    seed_all(seed)
    adapter=Adapter(cfg,method,variant)
    batch_size=batch_size or (12 if domain=='synthetic' else 6)
    if method=='u3net': batch_size=8 if domain=='synthetic' else 4
    micro_batch=1 if method=='restormer' and domain=='insar' else batch_size
    optimizer=torch.optim.Adam(adapter.model.parameters(),lr=cfg['optim']['lr'],
        weight_decay=cfg['optim']['weight_decay'],eps=cfg['optim']['eps'])
    scaler=torch.amp.GradScaler('cuda')
    start=0;best=math.inf;history=[]
    if (folder/'last.pth').exists():
        state=torch.load(folder/'last.pth',map_location='cuda',weights_only=False)
        if state['config']!=cfg: raise ValueError('Cannot resume checkpoint with a different protocol')
        if state['manifest_sha256']!=file_hash(OUT/'manifest.json'):
            raise ValueError('Cannot resume checkpoint with a different data manifest')
        adapter.model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
        scaler.load_state_dict(state['scaler']);start=state['epoch']+1;best=state['best'];history=state['history']
        if state.get('teacher'):
            import copy
            adapter.teacher=copy.deepcopy(adapter.model).eval().requires_grad_(False)
            adapter.teacher.load_state_dict(state['teacher'])
        del state
        torch.cuda.empty_cache()
    w,t=arrays(manifest['domains'][domain]['train'])
    vw,vt=arrays(manifest['domains'][domain]['val'])
    started=time.perf_counter()
    for epoch in range(start,epochs):
        seed_all(seed+epoch)
        phase_start=cfg['training']['distill_start_epoch']
        if method=='u3net' and epoch>=phase_start and adapter.teacher is None:
            import copy
            adapter.teacher=copy.deepcopy(adapter.model).eval().requires_grad_(False)
            optimizer=torch.optim.Adam(adapter.model.parameters(),lr=cfg['optim']['lr'])
        relative_epoch=epoch-phase_start if adapter.teacher is not None else epoch
        lr=cfg['optim']['lr']*(.99**relative_epoch if method=='u3net' else 1.)
        for group in optimizer.param_groups: group['lr']=lr
        adapter.model.train()
        gen=torch.Generator(device='cuda').manual_seed(seed*100000+epoch)
        indices=torch.randperm(len(w),generator=gen,device='cuda')
        choices=torch.tensor([1000,1000,1000,1000,1000,30,20,10,5,0],device='cuda')
        all_snr=choices[torch.randint(len(choices),(len(w),),device='cuda',generator=gen)]
        noisy_inputs=observed(w,all_snr,gen)
        losses=[];updates=0;epoch_start=time.perf_counter()
        for offset in range(0,len(w),batch_size):
            idx=indices[offset:offset+batch_size];wi=noisy_inputs[idx];ti=t[idx];snr=all_snr[idx]
            # Precomputed per epoch so every method sees the same observations even at different batch sizes.
            condition_snr=torch.where(snr>100,30,snr)
            optimizer.zero_grad(set_to_none=True)
            batch_loss=0.
            for chunk in range(0,len(idx),micro_batch):
                end=min(chunk+micro_batch,len(idx))
                with torch.autocast('cuda',dtype=torch.float16):
                    loss=adapter.loss(wi[chunk:end],ti[chunk:end],condition_snr[chunk:end],epoch)
                    loss=loss*((end-chunk)/len(idx))
                if not torch.isfinite(loss): raise RuntimeError(f'Nonfinite loss {domain} {method} epoch {epoch}')
                scaler.scale(loss).backward();batch_loss+=float(loss.detach())
            scaler.unscale_(optimizer)
            if method!='u3net':torch.nn.utils.clip_grad_norm_(adapter.model.parameters(),1.)
            old_scale=scaler.get_scale();scaler.step(optimizer);scaler.update()
            updates+=int(scaler.get_scale()>=old_scale);losses.append(batch_loss)
        adapter.model.eval();seed_all(100000+seed)
        validation=[]
        for offset in range(0,len(vw),micro_batch):
            with torch.autocast('cuda',dtype=torch.float16): p=adapter.predict(vw[offset:offset+micro_batch])
            errors=p-vt[offset:offset+micro_batch]
            k=torch.round((-errors/(2*math.pi)).flatten(1).median(1).values)[:,None,None,None]
            validation.extend((errors+k*2*math.pi).abs().mean((1,2,3)).cpu().tolist())
        val=float(np.mean(validation));duration=time.perf_counter()-epoch_start
        if not math.isfinite(val) or updates==0: raise RuntimeError('Validation nonfinite or no successful updates')
        # U3Net is selected only from its completed-method (distillation) phase.
        improved=val<best and (method!='u3net' or adapter.teacher is not None)
        if improved:best=val
        history.append({'epoch':epoch,'loss':float(np.mean(losses)),'val_aligned_mae':val,
                        'effective_batch_size':batch_size,'micro_batch_size':micro_batch,
                        'seconds':duration,'successful_updates':updates,'lr':lr,'distillation':adapter.teacher is not None})
        state={'model':adapter.model.state_dict(),'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),
               'epoch':epoch,'best':best,'history':history,'config':cfg,'seed':seed,'variant':variant,
               'manifest_sha256':file_hash(OUT/'manifest.json'),'teacher':adapter.teacher.state_dict() if adapter.teacher else None}
        temp=folder/'last.tmp.pth';torch.save(state,temp);temp.replace(folder/'last.pth')
        if improved:torch.save({'model':state['model'],'config':cfg,'epoch':epoch,'variant':variant},folder/'best.pth')
        dump(folder/'history.json',history)
        status('training',domain=domain,method=method,variant=variant,seed=seed,epoch=epoch+1,epochs=epochs,
               loss=history[-1]['loss'],validation_mae=val,epoch_seconds=duration)
    dump(folder/'complete.json',{'epochs':epochs,'best_validation_mae':best,'session_seconds':time.perf_counter()-started,
                                 'from_scratch':True,'seed':seed,'manifest_sha256':file_hash(OUT/'manifest.json')})


def load_adapter(domain,method,seed=42,variant='both'):
    folder=job_dir(domain,method,seed,variant)
    state=torch.load(folder/'best.pth',map_location='cpu',weights_only=False)
    adapter=Adapter(state['config'],method,variant)
    adapter.model.load_state_dict(state['model'],strict=True);adapter.model.eval()
    return adapter


def metric_row(p,t,w):
    p=p.astype(np.float64);t=t.astype(np.float64);w=w.astype(np.float64)
    row=metrics(p,t,w)
    e=p-t;offset=np.median(-e);centered=e+offset
    row.update(real_offset_mae=float(np.abs(centered).mean()),real_offset_rmse=float(np.sqrt(np.mean(centered**2))),
               abs_error_p95=float(np.percentile(np.abs(e),95)),abs_error_p99=float(np.percentile(np.abs(e),99)))
    gradient=np.hypot(*np.gradient(t));mask=gradient>=np.quantile(gradient,.9)
    k=row['global_offset_k'];row['high_gradient_mae']=float(np.abs(e+k*2*math.pi)[mask].mean())
    row['reference_cycle_mae']=float(np.abs(np.angle(np.exp(1j*(t-w)))).mean())
    return row


def evaluate(domain,method,seed=42,variant='both',real=False):
    import csv
    manifest=prepare();entries=manifest['real'] if real else manifest['domains'][domain]['test']
    label='real' if real else domain
    output=OUT/'evaluation'/f'{label}_{method}_{variant}_seed{seed}'
    if output.with_suffix('.json').exists():return
    adapter=None if method in METHODS else load_adapter(domain,method,seed,variant)
    rows=[];first={};histograms={};edges=np.concatenate((np.linspace(0,1,101),np.linspace(1.1,10,90),[20,50,100,1000]))
    for snr in ((None,) if real else SNRS):
        for i,entry in enumerate(entries):
            w,t=read_pair(entry)
            if snr is not None:
                rng=np.random.default_rng(42000+i+int(snr)*10000)
                w=np.angle(np.exp(1j*(w+rng.normal(0,math.sqrt(10**.1/10**(snr/10)),w.shape)))).astype(np.float32)
            if adapter:
                tensor=torch.from_numpy(w)[None,None].cuda()
                seed_all(42000+i)
                with torch.autocast('cuda',dtype=torch.float16):p=adapter.predict(tensor,snr=30 if snr is None else snr)
                p=p.squeeze().cpu().numpy()
            else:p=METHODS[method](w)
            if not np.isfinite(p).all():raise RuntimeError('Nonfinite prediction')
            rows.append({'sample':entry['id'],'snr':'clean' if snr is None else str(snr),**metric_row(p,t,w)})
            key='clean' if snr is None else str(snr)
            histograms.setdefault(key,np.zeros(len(edges)-1,dtype=np.int64))
            histograms[key]+=np.histogram(np.abs(p.astype(np.float64)-t),bins=edges)[0]
            if i==0:first[str(snr)]={'wrapped':w,'target':t,'prediction':p}
        status('evaluating',domain=label,method=method,snr=snr,variant=variant,seed=seed)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.with_suffix('.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary={}
    for snr in sorted(set(r['snr'] for r in rows)):
        chosen=[r for r in rows if r['snr']==snr]
        summary[snr]={k:float(np.mean([r[k] for r in chosen])) for k in chosen[0] if k not in ('sample','snr')}
        vals=np.asarray([r['aligned_mae'] for r in chosen]);rng=np.random.default_rng(42)
        boot=vals[rng.integers(0,len(vals),(1000,len(vals)))].mean(1)
        summary[snr]['aligned_mae_ci95']=[float(x) for x in np.quantile(boot,[.025,.975])]
    np.savez_compressed(str(output)+'_first.npz',**{f'{snr}_{k}':v for snr,d in first.items() for k,v in d.items()})
    dump(output.with_suffix('.json'),{'domain':label,'method':method,'seed':seed,'variant':variant,'samples':len(entries),
        'precision':'fp16 autocast / float32 outputs' if adapter else 'numpy CPU','draws':1,'summary':summary,
        'reference_warning':real or domain=='insar','source':'new_training' if adapter else 'label_free',
        'native_label':'clean means no additional injected noise; native InSAR observations are not noise-free',
        'error_histograms':{key:counts.tolist() for key,counts in histograms.items()},'error_histogram_edges':edges.tolist()})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','train','evaluate','queue'])
    parser.add_argument('--domain',choices=list(DOMAINS),default='synthetic')
    parser.add_argument('--method',default='fdu');parser.add_argument('--epochs',type=int,default=30)
    parser.add_argument('--seed',type=int,default=42);parser.add_argument('--variant',default='both',choices=['both','base','dcc','wwf'])
    parser.add_argument('--real',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=True
    if args.command=='prepare':prepare()
    elif args.command=='train':train(args.domain,args.method,args.epochs,args.seed,args.variant)
    elif args.command=='evaluate':evaluate(args.domain,args.method,args.seed,args.variant,args.real)
    else:
        import subprocess
        prepare();jobs=[]
        # All seven methods in both domains; U3Net's pilot budget is 50+20 epochs.
        for domain in DOMAINS:
            for method in DEEP:
                jobs.append(['train','--domain',domain,'--method',method,'--epochs','70' if method=='u3net' else '30'])
                jobs.append(['evaluate','--domain',domain,'--method',method])
                if domain=='insar':jobs.append(['evaluate','--domain',domain,'--method',method,'--real'])
            for method in METHODS:
                jobs.append(['evaluate','--domain',domain,'--method',method])
                if domain=='insar':jobs.append(['evaluate','--domain',domain,'--method',method,'--real'])
        for seed in (42,43,44):
            for variant in ('base','dcc','wwf','both'):
                if seed==42 and variant=='both':continue
                jobs.extend([['train','--domain','synthetic','--method','fdu','--epochs','30','--seed',str(seed),'--variant',variant],
                             ['evaluate','--domain','synthetic','--method','fdu','--seed',str(seed),'--variant',variant]])
        dump(OUT/'queue.json',jobs)
        for index,job in enumerate(jobs):
            status('job_started',index=index,total=len(jobs),command=job)
            with (OUT/'run.log').open('a',encoding='utf-8') as log:
                code=subprocess.call([sys.executable,'-B','-u',str(Path(__file__).resolve()),*job],cwd=ROOT,
                                     stdout=log,stderr=subprocess.STDOUT)
            if code:
                status('job_failed',index=index,total=len(jobs),command=job,exit_code=code)
                raise RuntimeError(f'Job failed with exit {code}: {job}')
        status('training_and_comparison_complete',jobs=len(jobs))


if __name__=='__main__':main()
