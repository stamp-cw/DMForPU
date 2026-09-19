"""Run the unmodified official Keras SQD-LSTM on GFS128 via a 256px data adapter."""
from __future__ import annotations

import argparse, csv, json, math, os, sys, time
from pathlib import Path

import h5py
import numpy as np
os.environ.setdefault("KERAS_BACKEND", "torch")
import keras
import torch
import torch.nn.functional as F
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam


ROOT=Path(__file__).resolve().parents[1]
UP=ROOT/"third_party"/"DeepPhaseUnwrap"
OUT=ROOT/"experiments"/"results"/"gfs128_upstream"/"runs"/"sqd_lstm"
# Keras 2 allowed AveragePooling2D() without pool_size; keep the official source
# untouched and supply that historical default at the compatibility boundary.
_AveragePooling2D = keras.layers.AveragePooling2D
class _Keras2AveragePooling2D(_AveragePooling2D):
    def __init__(self, pool_size=(2,2), *args, **kwargs): super().__init__(pool_size=pool_size, *args, **kwargs)
keras.layers.AveragePooling2D = _Keras2AveragePooling2D
import keras.backend as K
K.mean, K.abs, K.square = keras.ops.mean, keras.ops.abs, keras.ops.square

sys.path.insert(0,str(UP))
from src.models.architectures import JointConvSQDLSTMNet  # noqa: E402
from src.models.losses import tv_loss_plus_var_loss  # noqa: E402
sys.path.insert(0, str(ROOT))
from utils.phase_metrics import au_metrics_numpy, u3_aligned_metrics_numpy  # noqa: E402

SEED=42;BATCH=4;EPOCHS=100;SNRS=(0,5,10,20,30);TWO_PI=2*math.pi


def dump(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(path)


class DataSequence(keras.utils.Sequence):
    def __init__(self,w,t,ids,batch=BATCH,shuffle=True):self.w=w;self.t=t;self.ids=np.asarray(ids);self.batch=batch;self.shuffle=shuffle;self.order=self.ids.copy();self.on_epoch_end()
    def __len__(self):return math.ceil(len(self.ids)/self.batch)
    def __getitem__(self,i):
        ids=self.order[i*self.batch:(i+1)*self.batch]
        w=F.interpolate(torch.from_numpy(self.w[ids,None]),(256,256),mode='bilinear',align_corners=False).permute(0,2,3,1).numpy()
        t=F.interpolate(torch.from_numpy(self.t[ids,None]),(256,256),mode='bilinear',align_corners=False).permute(0,2,3,1).numpy()
        return w,t
    def on_epoch_end(self):
        if self.shuffle:np.random.default_rng(SEED+getattr(self,'epoch',0)).shuffle(self.order)
        self.epoch=getattr(self,'epoch',0)+1


class HistoryWriter(Callback):
    def __init__(self):super().__init__();self.rows=[];self.begin=0
    def on_epoch_begin(self,epoch,logs=None):self.begin=time.perf_counter()
    def on_epoch_end(self,epoch,logs=None):
        row={'epoch':epoch+1,'train_loss':float(logs['loss']),'val_loss':float(logs['val_loss']),'seconds':time.perf_counter()-self.begin}
        self.rows.append(row);dump(OUT/'history.json',self.rows)
        with (OUT/'history.csv.tmp').open('w',newline='',encoding='utf-8') as f:wr=csv.DictWriter(f,fieldnames=list(row));wr.writeheader();wr.writerows(self.rows)
        (OUT/'history.csv.tmp').replace(OUT/'history.csv');dump(OUT/'status.json',{'state':'training',**row,'epochs':EPOCHS})


def metrics(pred,target,wrapped):
    raw=pred-target;mean=raw-raw.mean((1,2),keepdims=True);span=np.ptp(target,axis=(1,2));integer=raw+np.round(np.median(-raw,(1,2))/TWO_PI)[:,None,None]*TWO_PI
    cycle=(np.mod(np.mod(pred+math.pi,TWO_PI)-math.pi-wrapped+math.pi,TWO_PI)-math.pi)
    values={'raw_mae':float(np.mean(np.abs(raw))),'integer_aligned_mae':float(np.mean(np.abs(integer))),
            'mean_aligned_mae':float(np.mean(np.abs(mean))),'mean_aligned_rmse':float(np.mean(np.sqrt(np.mean(mean**2,(1,2))))),
            'mean_aligned_nrmse':float(np.mean(np.sqrt(np.mean(mean**2,(1,2)))/np.maximum(span,1e-12))),
            'rewrap_circular_mae':float(np.mean(np.abs(cycle)))}
    values.update({key: float(value.mean()) for key, value in u3_aligned_metrics_numpy(pred, target).items()})
    values.update({key: float(value.mean()) for key, value in au_metrics_numpy(pred, target).items()})
    return values


def main(smoke=False):
    OUT.mkdir(parents=True,exist_ok=True);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED);keras.utils.set_random_seed(SEED)
    if not smoke and (OUT/'complete.json').exists():
        print('sqd_lstm complete',flush=True);return
    manifest=json.loads((ROOT/'experiments/results/gfs128_upstream/manifest.json').read_text(encoding='utf-8'))
    with h5py.File(ROOT/'data/GFS128/train.h5','r') as f:w=f['psi'][:];t=f['phi'][:]
    train=DataSequence(w,t,manifest['train_indices'],shuffle=True);val=DataSequence(w,t,manifest['validation_indices'],shuffle=False)
    model=JointConvSQDLSTMNet((256,256,1)).getModel();model.compile(optimizer=Adam(learning_rate=1e-3),loss=tv_loss_plus_var_loss)
    if smoke:
        x,y=train[0];torch.cuda.reset_peak_memory_stats();started=time.perf_counter();loss=model.train_on_batch(x,y);torch.cuda.synchronize()
        print(json.dumps({'parameters':model.count_params(),'shape':list(model(x[:1],training=False).shape),'loss':float(loss),'backend':keras.backend.backend(),
                          'step_seconds':time.perf_counter()-started,'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
                          'peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20}));return
    protocol={'method':'sqd_lstm','official_repo':'DeepPhaseUnwrap','official_commit':'df8bad82cdbde376c3516e8980eca0e0e9f80f75','official_source_modified':False,
              'input_adapter':'GFS128 bilinear 128->256 because official architecture/loss hard-code 256; prediction bilinear 256->128','batch':BATCH,'epochs':EPOCHS,'early_stopping_patience':10}
    dump(OUT/'protocol.json',protocol);writer=HistoryWriter()
    model.fit(train,validation_data=val,epochs=EPOCHS,callbacks=[writer,EarlyStopping(monitor='loss',patience=10,verbose=1),ModelCheckpoint(str(OUT/'best.keras'),monitor='loss',save_best_only=True)],verbose=2)
    model.save(OUT/'last.keras');result={'method':'sqd_lstm',**protocol,'parameters':model.count_params(),'tests':{}}
    for snr in SNRS:
        with h5py.File(ROOT/'data/GFS128'/f'test_{snr}dB.h5','r') as f:wi=f['psi'][:];ti=f['phi'][:]
        preds=[]
        for off in range(0,len(wi),BATCH):
            x=F.interpolate(torch.from_numpy(wi[off:off+BATCH,None]),(256,256),mode='bilinear',align_corners=False).permute(0,2,3,1)
            p=model(x,training=False).permute(0,3,1,2)
            p=F.interpolate(p,(128,128),mode='bilinear',align_corners=False)[:,0].detach().cpu().numpy();preds.append(p)
        result['tests'][str(snr)]=metrics(np.concatenate(preds),ti,wi)
    dump(OUT/'complete.json',result);dump(OUT/'status.json',{'state':'complete',**result})


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');args=ap.parse_args();main(args.smoke)
