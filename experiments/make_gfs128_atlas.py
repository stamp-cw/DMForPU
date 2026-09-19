"""Create and verify a visual atlas for the current GFS128 dataset."""
from __future__ import annotations

import json
import math
import shutil
from datetime import datetime
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib.pagesizes import A4, landscape
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/GFS128"
OUT = ROOT / "output/pdf"
TMP = ROOT / "tmp/pdfs/gfs128_atlas_pages"
PDF = OUT / "gfs128_dataset_atlas.pdf"
META = OUT / "gfs128_dataset_atlas.json"
CONDITIONS = ("clean", "0", "5", "10", "20", "30")
BG, INK, MUTED, ACCENT = "#f5f7fb", "#162235", "#617086", "#146ca4"


def setup_style():
    font_path = Path(r"C:\Windows\Fonts\msyh.ttc")
    if font_path.exists():
        fm.fontManager.addfont(font_path)
        font = fm.FontProperties(fname=font_path).get_name()
    else:
        font = "DejaVu Sans"
    plt.rcParams.update({"font.family": font, "axes.unicode_minus": False,
                         "figure.facecolor": BG, "axes.facecolor": "white",
                         "text.color": INK, "axes.labelcolor": INK, "axes.titlecolor": INK})


def path_for(condition):
    return DATA / ("test_clean.h5" if condition == "clean" else f"test_{condition}dB.h5")


def wrap(x):
    return np.remainder(x + math.pi, 2 * math.pi) - math.pi


def read_rows(path, keys, indices):
    order = np.argsort(indices); sorted_ids = np.asarray(indices)[order]
    with h5py.File(path, "r") as f:
        result = [f[key][sorted_ids] for key in keys]
    inverse = np.argsort(order)
    return [value[inverse] for value in result]


def frame(fig, section, page):
    fig.text(.035, .968, "DMForPU  |  GFS128 数据集图册", fontsize=9, weight="bold", color=MUTED)
    fig.text(.965, .968, section, fontsize=9, ha="right", color=MUTED)
    fig.add_artist(plt.Line2D([.035, .965], [.95, .95], transform=fig.transFigure, color="#d5dce7", lw=.8))
    fig.text(.035, .022, "固定样本索引、统一包裹相位色标；生成脚本：experiments/make_gfs128_atlas.py", fontsize=7.2, color=MUTED)
    fig.text(.965, .022, f"{page:02d}", fontsize=8, ha="right", color=MUTED)


def save_page(fig, page, slug, pages):
    target = TMP / f"{page:02d}_{slug}.png"
    fig.savefig(target, dpi=170, facecolor=BG)
    plt.close(fig); pages.append(target)


def compute_stats():
    train_counts = {}
    with h5py.File(DATA / "train.h5", "r") as f:
        snr = f["snr"][:]
        for c in CONDITIONS:
            train_counts[c] = int(np.isinf(snr).sum()) if c == "clean" else int(np.isclose(snr, float(c)).sum())
    spans, stds, grads = [], [], []
    with h5py.File(path_for("clean"), "r") as f:
        for begin in range(0, len(f["phi"]), 100):
            phi = f["phi"][begin:begin+100].astype(np.float64)
            spans.append(np.ptp(phi, axis=(1, 2))); stds.append(phi.std(axis=(1, 2)))
            gx, gy = np.diff(phi, axis=2), np.diff(phi, axis=1)
            grads.append(np.sqrt((np.mean(gx*gx, axis=(1, 2)) + np.mean(gy*gy, axis=(1, 2))) / 2))
    circular = {}; exact = {}; scene_ok = {}; target_max_diff = {}
    with h5py.File(path_for("clean"), "r") as ref:
        ref_phi = ref["phi"][:]; ref_ids = ref["scene_id"][:]
    for c in CONDITIONS:
        total = 0.; count = 0; maximum = 0.
        with h5py.File(path_for(c), "r") as f:
            scene_ok[c] = bool(np.array_equal(f["scene_id"][:], ref_ids))
            target_max_diff[c] = float(np.max(np.abs(f["phi"][:] - ref_phi)))
            for begin in range(0, len(f["phi"]), 100):
                phi = f["phi"][begin:begin+100]; psi = f["psi"][begin:begin+100]
                delta = wrap(psi - wrap(phi)); total += float(np.abs(delta).sum()); count += delta.size
                maximum = max(maximum, float(np.abs(delta).max()))
        circular[c] = total / count; exact[c] = maximum
    return {"train_counts": train_counts, "phase_span": np.concatenate(spans),
            "phase_std": np.concatenate(stds), "gradient_rms": np.concatenate(grads),
            "circular_mae": circular, "max_circular_error": exact,
            "scene_ids_match": scene_ok, "target_max_difference": target_max_diff}


def representative_indices(spans):
    order = np.argsort(spans)
    return [int(order[round(q*(len(order)-1))]) for q in (.05, .18, .34, .50, .66, .82, .95)]


def training_indices():
    rng = np.random.default_rng(20240913); result = []
    with h5py.File(DATA / "train.h5", "r") as f: snr = f["snr"][:]
    for c in CONDITIONS:
        candidates = np.flatnonzero(np.isinf(snr)) if c == "clean" else np.flatnonzero(np.isclose(snr, float(c)))
        result.append(int(rng.choice(candidates)))
    return result


def cover(stats, pages):
    qa = json.loads((DATA / "qa_report.json").read_text(encoding="utf-8"))
    fig = plt.figure(figsize=(11.69, 8.27)); ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.add_patch(plt.Rectangle((0, .60), 1, .40, color=INK, transform=ax.transAxes))
    fig.text(.065, .84, "GFS128", fontsize=42, color="white", weight="bold")
    fig.text(.067, .75, "相位解缠数据集图册", fontsize=25, color="white", weight="bold")
    fig.text(.068, .69, "Current Dataset Atlas  |  Generator v2.0", fontsize=12, color="#c6d8e9")
    rows = [("空间尺寸", "128 x 128"), ("训练集", "5,000 张；clean、0、5、10、20、30 dB 混合"),
            ("测试集", "6 个匹配测试集，每种条件 1,000 张"),
            ("字段", "phi：无噪声展开相位；psi：含噪或无噪包裹相位"),
            ("场景隔离", "训练与测试无重复场景"), ("自动质量检查", "全部通过" if qa["checks"]["all_pass"] else "存在失败项")]
    y = .52
    for key, value in rows:
        fig.text(.07, y, key, fontsize=10, weight="bold", color=MUTED)
        fig.text(.20, y, value, fontsize=11, color=INK); y -= .062
    fig.text(.62, .51, "训练条件数量", fontsize=14, weight="bold")
    y = .455
    for condition, count in stats["train_counts"].items():
        fig.text(.64, y, f"{condition:>5}", fontsize=10, family="monospace")
        fig.text(.75, y, f"{count:,}", fontsize=10, weight="bold"); y -= .047
    fig.text(.93, .05, datetime.now().strftime("%Y-%m-%d %H:%M"), ha="right", fontsize=8.5, color=MUTED)
    save_page(fig, 1, "cover", pages)


def clean_gallery(ids, pages):
    phi, = read_rows(path_for("clean"), ("phi",), ids)
    fig, axes = plt.subplots(2, 4, figsize=(11.69, 8.27), gridspec_kw={"top":.79,"bottom":.09,"left":.055,"right":.96,"hspace":.28,"wspace":.14})
    fig.suptitle("无噪声展开相位形态", x=.04, y=.905, ha="left", fontsize=21, weight="bold", color=ACCENT)
    fig.text(.04, .845, "按相位跨度分位数选择代表场景；每幅图使用独立展开相位色标。", fontsize=9.5, color=MUTED)
    for ax, image, idx in zip(axes.flat, phi, ids):
        im=ax.imshow(image,cmap="turbo"); ax.set_title(f"scene {idx:04d}  |  span {np.ptp(image):.1f} rad",fontsize=8.5)
        ax.set_xticks([]);ax.set_yticks([]);plt.colorbar(im,ax=ax,fraction=.044,pad=.025)
    axes.flat[-1].axis("off"); frame(fig,"Clean unwrapped phase",2); save_page(fig,2,"clean_phase_gallery",pages)


def train_pairs(ids, pages):
    phi, psi, snr = read_rows(DATA/"train.h5", ("phi","psi","snr"), ids)
    fig, axes=plt.subplots(3,4,figsize=(11.69,8.27),gridspec_kw={"top":.79,"bottom":.09,"left":.055,"right":.965,"hspace":.26,"wspace":.09})
    fig.suptitle("训练集六种条件样本对",x=.04,y=.905,ha="left",fontsize=21,weight="bold",color=ACCENT)
    fig.text(.04,.845,"每行两组样本；左侧为无噪声展开标签 phi，右侧为训练输入 psi。",fontsize=9.5,color=MUTED)
    for k,(idx,gt,obs,s) in enumerate(zip(ids,phi,psi,snr)):
        row,col=divmod(k,2); left=col*2; label="clean" if np.isinf(s) else f"{s:g} dB"
        axes[row,left].imshow(gt,cmap="turbo");axes[row,left+1].imshow(obs,cmap="twilight_shifted",vmin=-math.pi,vmax=math.pi)
        axes[row,left].set_title(f"ID {idx:04d} | phi",fontsize=8.5);axes[row,left+1].set_title(f"{label} | psi",fontsize=8.5)
    for ax in axes.flat:ax.set_xticks([]);ax.set_yticks([])
    frame(fig,"Training pairs",3);save_page(fig,3,"training_pairs",pages)


def matched_gallery(ids, pages):
    phi, = read_rows(path_for("clean"),("phi",),ids)
    observations={c:read_rows(path_for(c),("psi",),ids)[0] for c in CONDITIONS}
    fig,axes=plt.subplots(len(ids),7,figsize=(11.69,8.27),gridspec_kw={"top":.79,"bottom":.08,"left":.04,"right":.975,"hspace":.08,"wspace":.035})
    fig.suptitle("同一测试场景的匹配噪声条件",x=.04,y=.905,ha="left",fontsize=20,weight="bold",color=ACCENT)
    fig.text(.04,.845,"每行 scene_id 完全相同；第一列是展开标签，其余列统一使用 [-pi, pi] 包裹相位色标。",fontsize=9.3,color=MUTED)
    headers=("phi",*CONDITIONS)
    for j,h in enumerate(headers):axes[0,j].set_title(h if h=="clean" else (h if h=="phi" else f"{h} dB"),fontsize=8.5,weight="bold")
    for i,idx in enumerate(ids):
        axes[i,0].imshow(phi[i],cmap="turbo");axes[i,0].set_ylabel(f"{idx:04d}",fontsize=7.5,rotation=0,labelpad=16)
        for j,c in enumerate(CONDITIONS,1):axes[i,j].imshow(observations[c][i],cmap="twilight_shifted",vmin=-math.pi,vmax=math.pi)
    for ax in axes.flat:ax.set_xticks([]);ax.set_yticks([])
    frame(fig,"Matched test conditions",4);save_page(fig,4,"matched_conditions",pages)


def noise_page(index, stats, pages):
    phi,=read_rows(path_for("clean"),("phi",),[index]); truth=phi[0]; clean_wrap=wrap(truth)
    obs={c:read_rows(path_for(c),("psi",),[index])[0][0] for c in CONDITIONS}
    fig,axes=plt.subplots(2,6,figsize=(11.69,8.27),gridspec_kw={"top":.78,"bottom":.10,"left":.045,"right":.97,"hspace":.24,"wspace":.08})
    fig.suptitle(f"噪声作用细节：scene {index:04d}",x=.04,y=.905,ha="left",fontsize=20,weight="bold",color=ACCENT)
    fig.text(.04,.845,"上排：包裹相位；下排：相对于无噪声包裹相位的圆周误差。",fontsize=9.5,color=MUTED)
    for j,c in enumerate(CONDITIONS):
        axes[0,j].imshow(obs[c],cmap="twilight_shifted",vmin=-math.pi,vmax=math.pi);axes[0,j].set_title(c if c=="clean" else f"{c} dB",fontsize=9,weight="bold")
        delta=wrap(obs[c]-clean_wrap);im=axes[1,j].imshow(delta,cmap="coolwarm",vmin=-math.pi,vmax=math.pi)
        axes[1,j].set_title(f"MAE {np.abs(delta).mean():.3f}",fontsize=8)
    for ax in axes.flat:ax.set_xticks([]);ax.set_yticks([])
    frame(fig,"Noise detail",5);save_page(fig,5,"noise_detail",pages)


def statistics(stats,pages):
    fig,axes=plt.subplots(2,2,figsize=(11.69,8.27),gridspec_kw={"top":.79,"bottom":.11,"left":.08,"right":.96,"hspace":.36,"wspace":.28})
    fig.suptitle("数据分布与质量统计",x=.04,y=.905,ha="left",fontsize=21,weight="bold",color=ACCENT)
    fig.text(.04,.845,"统计基于 1,000 个固定测试场景；误差使用圆周距离。",fontsize=9.5,color=MUTED)
    axes[0,0].bar(range(6),[stats["train_counts"][c] for c in CONDITIONS],color=ACCENT,alpha=.85);axes[0,0].set_xticks(range(6),CONDITIONS);axes[0,0].set_title("训练集条件数量");axes[0,0].set_ylabel("images")
    axes[0,1].plot(range(6),[stats["circular_mae"][c] for c in CONDITIONS],marker="o",color="#ce5b3e");axes[0,1].set_xticks(range(6),CONDITIONS);axes[0,1].set_title("观测噪声圆周 MAE");axes[0,1].set_ylabel("rad")
    axes[1,0].hist(stats["phase_span"],bins=30,color=ACCENT,alpha=.82);axes[1,0].set_title("展开相位跨度分布");axes[1,0].set_xlabel("max(phi)-min(phi), rad");axes[1,0].set_ylabel("scenes")
    axes[1,1].hist(stats["gradient_rms"],bins=30,color="#5a9c6f",alpha=.85);axes[1,1].set_title("空间梯度 RMS 分布");axes[1,1].set_xlabel("rad/pixel");axes[1,1].set_ylabel("scenes")
    for ax in axes.flat:ax.grid(alpha=.2)
    frame(fig,"Statistics and QA",6);save_page(fig,6,"statistics",pages)


def assemble(pages):
    width,height=landscape(A4);c=canvas.Canvas(str(PDF),pagesize=(width,height),pageCompression=1)
    c.setTitle("GFS128 Dataset Atlas");c.setAuthor("DMForPU")
    for page in pages:
        c.drawImage(str(page),0,0,width=width,height=height,preserveAspectRatio=True,anchor="c");c.showPage()
    c.save()


def main():
    setup_style();OUT.mkdir(parents=True,exist_ok=True)
    if TMP.exists():shutil.rmtree(TMP)
    TMP.mkdir(parents=True)
    stats=compute_stats();ids=representative_indices(stats["phase_span"]);train_ids=training_indices();pages=[]
    cover(stats,pages);clean_gallery(ids,pages);train_pairs(train_ids,pages);matched_gallery(ids[:4],pages)
    noise_page(ids[len(ids)//2],stats,pages);statistics(stats,pages);assemble(pages)
    meta={"title":"GFS128 Dataset Atlas","created":datetime.now().isoformat(timespec="seconds"),
          "generator":"experiments/make_gfs128_atlas.py","pdf":str(PDF.relative_to(ROOT)),"pages":len(pages),
          "representative_test_indices":ids,"training_indices":train_ids,
          "train_condition_counts":stats["train_counts"],"circular_mae":stats["circular_mae"],
          "max_circular_error":stats["max_circular_error"],"scene_ids_match":stats["scene_ids_match"],
          "target_max_difference":stats["target_max_difference"]}
    META.write_text(json.dumps(meta,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    print(PDF);print(META);print(TMP)


if __name__=="__main__":main()
