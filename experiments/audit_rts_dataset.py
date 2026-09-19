"""Full-file QA and statistics for an already generated RTS dataset."""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/"data";SIZES=(128,64,32);SNRS=(0,5,10,20,30)
PIXEL_FIELDS=("psi","phi","phi_wrapped_clean","coherence","dem","deformation_phase","noise_map")


def dump(path,value):
    tmp=path.with_suffix(path.suffix+".tmp");tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8");tmp.replace(path)


def wrap(x):return np.angle(np.exp(1j*x))


def audit_file(path:Path,expected:int,collect=False):
    result={"samples":expected,"finite":True,"psi_min":math.inf,"psi_max":-math.inf,"phi_min":math.inf,"phi_max":-math.inf,"coherence_min":math.inf,"coherence_max":-math.inf,"max_clean_wrap_error":0.,"max_noise_map_error":0.}
    distributions={"phase_span":[],"coherence_mean":[],"dem_range":[],"phase_gradient_mean":[]}
    with h5py.File(path,"r") as f:
        missing=[k for k in PIXEL_FIELDS if k not in f]
        if missing:raise RuntimeError(f"{path}: missing {missing}")
        if any(f[k].shape!=(expected,int(f.attrs["image_size"]),int(f.attrs["image_size"])) for k in PIXEL_FIELDS):raise RuntimeError(f"{path}: shape mismatch")
        for begin in range(0,expected,64):
            end=min(expected,begin+64);values={k:f[k][begin:end].astype(np.float32) for k in PIXEL_FIELDS}
            result["finite"] &= all(np.isfinite(x).all() for x in values.values())
            result["psi_min"]=min(result["psi_min"],float(values["psi"].min()));result["psi_max"]=max(result["psi_max"],float(values["psi"].max()))
            result["phi_min"]=min(result["phi_min"],float(values["phi"].min()));result["phi_max"]=max(result["phi_max"],float(values["phi"].max()))
            result["coherence_min"]=min(result["coherence_min"],float(values["coherence"].min()));result["coherence_max"]=max(result["coherence_max"],float(values["coherence"].max()))
            result["max_clean_wrap_error"]=max(result["max_clean_wrap_error"],float(np.max(np.abs(wrap(values["phi"])-values["phi_wrapped_clean"]))))
            result["max_noise_map_error"]=max(result["max_noise_map_error"],float(np.max(np.abs(wrap(values["psi"]-values["phi_wrapped_clean"])-values["noise_map"]))))
            if collect:
                distributions["phase_span"].extend(np.ptp(values["phi"],axis=(1,2)).tolist());distributions["coherence_mean"].extend(np.mean(values["coherence"],axis=(1,2)).tolist());distributions["dem_range"].extend(np.ptp(values["dem"],axis=(1,2)).tolist())
                gx=np.diff(values["phi"],axis=2);gy=np.diff(values["phi"],axis=1);distributions["phase_gradient_mean"].extend(((np.mean(np.abs(gx),axis=(1,2))+np.mean(np.abs(gy),axis=(1,2)))/2).tolist())
        for key in ("partition","region_id","difficulty","deformation_type","scene_mode","satellite_id","snr"):
            vals=f[key][:];result[key+"_counts"]={str(float(k) if key=="snr" else int(k)):int(v) for k,v in zip(*np.unique(vals,return_counts=True))}
        result["metadata_maps"]=json.loads(f.attrs["metadata_maps"]);result["region_maps"]=json.loads(f.attrs["region_maps"]);result["satellite_maps"]=json.loads(f.attrs["satellite_maps"])
    result["valid_ranges"]=bool(result["finite"] and result["psi_min"]>=-math.pi-1e-6 and result["psi_max"]<math.pi+1e-6 and result["phi_min"]>=-14*math.pi-1e-4 and result["phi_max"]<=14*math.pi+1e-4 and result["coherence_min"]>0 and result["coherence_max"]<=1 and result["max_clean_wrap_error"]<2e-5 and result["max_noise_map_error"]<2e-3)
    return result,distributions


def invariant_checks():
    checks={"cross_snr":{},"cross_size_train_metadata":{}}
    for size in SIZES:
        base=DATA/f"RTS{size}"/"test_0dB.h5";maximum=0.;metadata=True
        with h5py.File(base,"r") as a:
            for condition in ["clean", *(f"{snr}dB" for snr in SNRS[1:])]:
                with h5py.File(DATA/f"RTS{size}"/f"test_{condition}.h5","r") as b:
                    for begin in range(0,1000,100):maximum=max(maximum,float(np.max(np.abs(a["phi"][begin:begin+100]-b["phi"][begin:begin+100]))))
                    metadata &= all(np.array_equal(a[k][:],b[k][:]) for k in ("random_seed","region_id","partition","difficulty","deformation_type","scene_mode","satellite_id","source_row","source_col"))
        with h5py.File(DATA/f"RTS{size}"/"test_clean.h5","r") as clean:
            clean_exact=bool(np.isposinf(clean["snr"][:]).all() and np.count_nonzero(clean["noise_map"][:])==0 and np.array_equal(clean["psi"][:],clean["phi_wrapped_clean"][:]))
        checks["cross_snr"][str(size)]={"max_gt_difference":maximum,"metadata_identical":bool(metadata),"clean_exact":clean_exact}
    with h5py.File(DATA/"RTS128"/"train.h5","r") as base:
        for size in (64,32):
            with h5py.File(DATA/f"RTS{size}"/"train.h5","r") as other:
                checks["cross_size_train_metadata"][str(size)]={k:bool(np.array_equal(base[k][:],other[k][:])) for k in ("snr","random_seed","region_id","partition","difficulty","deformation_type","scene_mode","satellite_id","source_row","source_col")}
    return checks


def plot(distributions,category_result):
    fig,axes=plt.subplots(2,3,figsize=(14,8),constrained_layout=True)
    labels=(("phase_span","Phase span (rad)"),("coherence_mean","Mean coherence"),("dem_range","DEM range (m)"),("phase_gradient_mean","Mean |phase gradient| (rad/pixel)"))
    for ax,(key,label) in zip(axes.flat[:4],labels):ax.hist(distributions[key],bins=45,color="#2878b5",alpha=.82);ax.set_xlabel(label);ax.set_ylabel("Samples");ax.grid(alpha=.2)
    maps=category_result["metadata_maps"]
    for ax,key,title,mapping in ((axes[1,1],"difficulty_counts","Difficulty",maps["difficulty"]),(axes[1,2],"deformation_type_counts","Deformation type",maps["deformation_type"])):
        inverse={v:k for k,v in mapping.items()};counts=category_result[key];names=[inverse[int(k)] for k in counts];values=list(counts.values());ax.bar(range(len(names)),values,color="#d56a32");ax.set_xticks(range(len(names)),names,rotation=28,ha="right");ax.set_ylabel("Samples");ax.set_title(title);ax.grid(axis="y",alpha=.2)
    fig.suptitle("RTS128 training-set statistics");path=DATA/"RTS_statistics.png";fig.savefig(path,dpi=180);plt.close(fig);return path


def main():
    files={};distributions=None;category=None
    for size in SIZES:
        files[str(size)]={}
        for name,count in [("train",5000),("test_clean",1000),*((f"test_{s}dB",1000) for s in SNRS)]:
            result,dist=audit_file(DATA/f"RTS{size}"/f"{name}.h5",count,collect=size==128 and name=="train");files[str(size)][name]=result
            if size==128 and name=="train":distributions=dist;category=result
            print(size,name,result["valid_ranges"],flush=True)
    invariants=invariant_checks();all_valid=all(x["valid_ranges"] for group in files.values() for x in group.values()) and all(x["max_gt_difference"]==0 and x["metadata_identical"] and x["clean_exact"] for x in invariants["cross_snr"].values()) and all(all(x.values()) for x in invariants["cross_size_train_metadata"].values())
    figure=plot(distributions,category);stats={"state":"pass" if all_valid else "fail","files":files,"invariants":invariants,"figure":str(figure.relative_to(ROOT)),"train_distribution_summary":{k:{"mean":float(np.mean(v)),"std":float(np.std(v)),"min":float(np.min(v)),"max":float(np.max(v))} for k,v in distributions.items()}}
    dump(DATA/"RTS_STATISTICS.json",stats)
    for manifest_path in (DATA/"RTS_DATASET.json",DATA/"DATASETS_V2.json"):
        obj=json.loads(manifest_path.read_text(encoding="utf-8"))
        target=obj if manifest_path.name=="RTS_DATASET.json" else obj["families"]["RTS"]
        target["full_audit"]={"state":stats["state"],"file":"data/RTS_STATISTICS.json","figure":"data/RTS_statistics.png"};dump(manifest_path,obj)
    if not all_valid:raise RuntimeError("RTS full audit failed")


if __name__=="__main__":main()
