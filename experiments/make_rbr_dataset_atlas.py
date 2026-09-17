"""Create the RBR dataset atlas from the released H5 files and GIS vectors."""
from __future__ import annotations

import json
import math
from datetime import date
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import numpy as np
import rasterio
from rasterio.windows import from_bounds
import shapefile
import yaml
from reportlab.lib.pagesizes import A4, landscape
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "pdf"
PAGES = OUT / "rbr_dataset_atlas_pages"
PDF = OUT / "rbr_dataset_atlas.pdf"
META = OUT / "rbr_dataset_atlas.json"
REGION_DIR = ROOT / "data" / "rbr_sources" / "regions"
SNRS = (30, 20, 10, 5, 0)
BG, INK, MUTED = "#f4f6fa", "#17243a", "#607086"
REGION_COLORS = ["#2d6a9f", "#c56a32", "#4f8b57", "#8b5b9e", "#b23a48"]
SPLIT_COLORS = {
    "train": "#2878b5", "validation": "#e6a532",
    "test_in_domain": "#4d9d66", "test_cross_region": "#bd4650",
}
CN_NAMES = {
    "qilian_mountains": "祁连山",
    "lanzhou_loess": "兰州—定西黄土高原",
    "north_china_plain": "华北平原",
    "huainan_mining": "淮南矿区",
    "sichuan_yunnan_holdout": "川滇植被山区",
}
DEFORMATION_NAMES = ["仅地形", "高斯沉降盆", "椭圆沉降盆", "多中心形变", "线性梯度", "非线性平滑", "局部高梯度"]


def setup_style() -> None:
    font = Path(r"C:\Windows\Fonts\msyh.ttc")
    if font.exists():
        fm.fontManager.addfont(font)
        family = fm.FontProperties(fname=font).get_name()
    else:
        family = "DejaVu Sans"
    plt.rcParams.update({
        "font.family": family, "axes.unicode_minus": False, "figure.facecolor": BG,
        "axes.facecolor": "white", "text.color": INK, "axes.labelcolor": INK,
        "axes.titlecolor": INK, "savefig.facecolor": BG,
    })


def add_frame(fig: plt.Figure, section: str, number: int) -> None:
    fig.text(0.035, 0.969, "DMForPU · RBR 数据集图册", fontsize=9, weight="bold", color=MUTED)
    fig.text(0.965, 0.969, section, fontsize=9, ha="right", color=MUTED)
    fig.add_artist(plt.Line2D([0.035, 0.965], [0.951, 0.951], transform=fig.transFigure, color="#d5dce7", lw=0.8))
    fig.text(0.035, 0.022, "正式数据：data/RBR{128,64,32} · 固定样本索引与统一色标", fontsize=7.4, color=MUTED)
    fig.text(0.965, 0.022, f"{number:02d}", fontsize=8, ha="right", color=MUTED)


def save_page(fig: plt.Figure, number: int, slug: str, paths: list[Path]) -> None:
    path = PAGES / f"{number:02d}_{slug}.png"
    fig.savefig(path, dpi=180, pad_inches=0)
    plt.close(fig)
    paths.append(path)


def read_h5(size: int, split: str, keys: list[str], ids: list[int]) -> dict[str, np.ndarray]:
    order = np.argsort(ids)
    sorted_ids = np.asarray(ids)[order]
    inv = np.argsort(order)
    path = ROOT / "data" / f"RBR{size}" / f"{split}.h5"
    with h5py.File(path, "r") as f:
        return {key: f[key][sorted_ids][inv] for key in keys}


def phase_spans() -> np.ndarray:
    path = ROOT / "data" / "RBR128" / "test_30dB.h5"
    with h5py.File(path, "r") as f:
        result = np.empty(len(f["phi"]), np.float32)
        for start in range(0, len(result), 100):
            values = f["phi"][start:start + 100]
            result[start:start + len(values)] = np.ptp(values, axis=(1, 2))
    return result


def representatives(spans: np.ndarray, count: int) -> list[int]:
    ranked = np.argsort(spans)
    return [int(ranked[round(q * (len(ranked) - 1))]) for q in np.linspace(0.03, 0.97, count)]


def ids_by_region(split: str = "test_10dB") -> list[int]:
    with h5py.File(ROOT / "data" / "RBR128" / f"{split}.h5", "r") as f:
        regions = f["region_id"][:]
        spans = np.ptp(f["phi"][:], axis=(1, 2))
    ids = []
    for rid in range(5):
        candidates = np.flatnonzero(regions == rid)
        median = np.median(spans[candidates])
        ids.append(int(candidates[np.argmin(np.abs(spans[candidates] - median))]))
    return ids


def cover_page(stats: dict, selected: dict, paths: list[Path]) -> None:
    fig = plt.figure(figsize=(11.69, 8.27)); ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.add_patch(Rectangle((0, 0.62), 1, 0.38, transform=ax.transAxes, color=INK))
    fig.text(0.065, 0.855, "RBR", fontsize=39, weight="bold", color="white")
    fig.text(0.065, 0.765, "真实地形驱动相位解缠数据集图册", fontsize=25, weight="bold", color="white")
    fig.text(0.067, 0.700, "Real-terrain-based Phase Unwrapping Dataset Atlas", fontsize=12.5, color="#c9d8ea")
    fig.text(0.065, 0.552, "数据协议", fontsize=15, weight="bold")
    rows = [
        ("地形来源", "Copernicus GLO-30 真实 DSM；其余相位与噪声由物理模型生成"),
        ("研究区", "4 个域内区域 + 1 个川滇跨区域留出区"),
        ("分辨率", "128×128、64×64、32×32，场景与元数据跨尺度对齐"),
        ("训练/验证", "5,000 场景：4,500 训练 + 500 验证"),
        ("测试", "每个尺度、每个 SNR 1,000 场景：700 域内 + 300 跨区域"),
        ("噪声档位", "训练：0/5/10/20/30/60 dB；测试：0/5/10/20/30 dB"),
    ]
    y = 0.505
    for label, value in rows:
        fig.text(0.07, y, label, fontsize=9.4, weight="bold", color=MUTED)
        fig.text(0.20, y, value, fontsize=10.1)
        y -= 0.052
    fig.text(0.64, 0.552, "数据质检", fontsize=15, weight="bold")
    facts = [
        "18 个 H5 文件全部通过审计",
        "跨 SNR 真值和元数据完全一致",
        "跨尺度场景、随机种子和空间来源一致",
        f"训练相位跨度均值 {stats['phase_span']['mean']:.2f} rad",
        f"训练 DEM 高差均值 {stats['dem_range']['mean']:.1f} m",
    ]
    for i, value in enumerate(facts):
        fig.text(0.65, 0.505 - i * 0.052, "●", color="#3a9160", fontsize=9)
        fig.text(0.675, 0.505 - i * 0.052, value, fontsize=9.7)
    fig.text(0.065, 0.075, f"代表样本 ID：{selected['clean']}", fontsize=8.5, color=MUTED)
    fig.text(0.93, 0.045, date.today().isoformat(), ha="right", fontsize=8.5, color=MUTED)
    save_page(fig, 1, "cover", paths)


def region_page(regions: list[dict], paths: list[Path]) -> None:
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.suptitle("研究区与空间独立划分", x=0.04, y=0.905, ha="left", fontsize=19, weight="bold")
    fig.text(0.04, 0.845, "训练、验证和域内测试按经度带留出；川滇区域完全不参与训练，用作跨区域测试。", fontsize=9.6, color=MUTED)
    ax = fig.add_axes([0.06, 0.13, 0.48, 0.64])
    for i, region in enumerate(regions):
        xmin, ymin, xmax, ymax = region["bbox"]
        ax.add_patch(Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, facecolor=REGION_COLORS[i], alpha=0.18, edgecolor=REGION_COLORS[i], lw=2))
        ax.scatter((xmin+xmax)/2, (ymin+ymax)/2, s=40, color=REGION_COLORS[i], zorder=3)
        ax.annotate(f"{i}  {CN_NAMES[region['name']]}", ((xmin+xmax)/2, (ymin+ymax)/2), xytext=(5, 5), textcoords="offset points", fontsize=8.3, weight="bold")
    ax.set_xlim(96.5, 119); ax.set_ylim(25.5, 41); ax.set_xlabel("经度 (°E)"); ax.set_ylabel("纬度 (°N)"); ax.grid(alpha=0.22)
    ax.set_title("五个 Copernicus GLO-30 地形来源区", fontsize=11, loc="left", pad=9)
    subaxes = []
    for i, region in enumerate(regions):
        row, col = divmod(i, 2)
        subaxes.append(fig.add_axes([0.59 + col*0.195, 0.60-row*0.205, 0.165, 0.135]))
    for i, (region, sax) in enumerate(zip(regions, subaxes)):
        sax.set_xlim(0, 1); sax.set_ylim(0, 1); sax.set_xticks([]); sax.set_yticks([])
        if region["role"] == "test_cross_region":
            sax.add_patch(Rectangle((0, 0), 1, 1, color=SPLIT_COLORS["test_cross_region"], alpha=0.75))
            caption = "跨区域测试 100%"
        else:
            for name, start, stop in (("train",0,.64),("validation",.68,.82),("test_in_domain",.84,1)):
                sax.add_patch(Rectangle((start, 0), stop-start, 1, color=SPLIT_COLORS[name], alpha=0.78))
            caption = "训练 / 验证 / 域内测试"
        sax.set_title(f"{i}  {CN_NAMES[region['name']]}", fontsize=8.4, loc="left", pad=3)
        sax.text(.5, -.20, caption, ha="center", transform=sax.transAxes, fontsize=7, color=MUTED)
    handles = [Patch(color=color, label=label) for label, color in [("训练",SPLIT_COLORS['train']),("验证",SPLIT_COLORS['validation']),("域内测试",SPLIT_COLORS['test_in_domain']),("跨区域测试",SPLIT_COLORS['test_cross_region'])]]
    fig.legend(handles=handles, loc="lower right", bbox_to_anchor=(.95,.12), ncol=2, frameon=False, fontsize=8)
    fig.text(.59,.105,"矢量文件包含完整区域边界及上述 13 个分区面，CRS 为 EPSG:4326。",fontsize=8,color=MUTED)
    add_frame(fig, "Regions and spatial splits", 2); save_page(fig, 2, "regions_and_splits", paths)


def dem_page(regions: list[dict], paths: list[Path]) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(11.69, 8.27), gridspec_kw={"top":.78,"bottom":.09,"left":.05,"right":.97,"hspace":.26,"wspace":.16})
    fig.suptitle("真实地形来源区", x=.04, y=.905, ha="left", fontsize=19, weight="bold")
    fig.text(.04,.845,"每幅图直接裁剪自下载的 Copernicus GLO-30 GeoTIFF；色标范围按区域独立设置。",fontsize=9.6,color=MUTED)
    for i, (region, ax) in enumerate(zip(regions, axes.flat)):
        tif = ROOT / "data" / "rbr_sources" / "copernicus_glo30" / f"{region['name']}.tif"
        with rasterio.open(tif) as src:
            window = from_bounds(*region["bbox"], transform=src.transform)
            data = src.read(1, window=window, out_shape=(420,420), masked=True)
        im=ax.imshow(data,cmap="terrain"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{i}  {CN_NAMES[region['name']]} · {region['scene_type']}",fontsize=9.2,loc="left")
        cb=fig.colorbar(im,ax=ax,fraction=.043,pad=.02); cb.ax.tick_params(labelsize=6.5); cb.set_label("m",fontsize=7)
    axes.flat[-1].axis("off")
    axes.flat[-1].text(.05,.88,"区域设置",fontsize=14,weight="bold")
    axes.flat[-1].text(.05,.58,"域内：祁连山、黄土高原、\n华北平原、淮南矿区\n\n跨域：川滇植被山区",fontsize=10,linespacing=1.6)
    axes.flat[-1].text(.05,.18,"地形是真实遥感产品；干涉相位、\n形变与相干噪声为可复现实验模拟。",fontsize=8.5,color=MUTED,linespacing=1.5)
    add_frame(fig,"Copernicus GLO-30 terrain",3); save_page(fig,3,"real_terrain",paths)


def clean_page(ids: list[int], paths: list[Path]) -> None:
    values=read_h5(128,"test_30dB",["phi","region_id"],ids)
    fig,axes=plt.subplots(3,4,figsize=(11.69,8.27),gridspec_kw={"top":.78,"bottom":.08,"left":.05,"right":.97,"hspace":.28,"wspace":.10})
    fig.suptitle("RBR128 干净展开相位形态",x=.04,y=.905,ha="left",fontsize=19,weight="bold")
    fig.text(.04,.845,"按相位跨度分位数固定选择 12 个测试场景；每图独立色标以呈现空间结构。",fontsize=9.6,color=MUTED)
    for ax,img,idx,rid in zip(axes.flat,values["phi"],ids,values["region_id"]):
        ax.imshow(img,cmap="turbo"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"ID {idx:04d} · 区域 {int(rid)} · Δφ={np.ptp(img):.1f} rad",fontsize=7.8)
    add_frame(fig,"Clean unwrapped phase",4); save_page(fig,4,"clean_phase_gallery",paths)


def snr_page(ids: list[int], paths: list[Path]) -> None:
    gt=read_h5(128,"test_30dB",["phi","region_id"],ids)
    noisy={snr:read_h5(128,f"test_{snr}dB",["psi"],ids)["psi"] for snr in SNRS}
    fig,axes=plt.subplots(5,6,figsize=(11.69,8.27),gridspec_kw={"top":.78,"bottom":.08,"left":.045,"right":.975,"hspace":.16,"wspace":.05})
    fig.suptitle("同一场景在不同 SNR 下的观测",x=.04,y=.905,ha="left",fontsize=19,weight="bold")
    fig.text(.04,.845,"每行来自一个区域；测试集在所有 SNR 文件中共享完全相同的真值和元数据。",fontsize=9.6,color=MUTED)
    for c,title in enumerate(["展开真值"]+[f"{x} dB" for x in SNRS]): axes[0,c].set_title(title,fontsize=9,weight="bold")
    for r,(idx,img,rid) in enumerate(zip(ids,gt["phi"],gt["region_id"])):
        axes[r,0].imshow(img,cmap="turbo"); axes[r,0].set_ylabel(f"区域 {int(rid)}\nID {idx:04d}",fontsize=7.5)
        for c,snr in enumerate(SNRS,1): axes[r,c].imshow(noisy[snr][r],cmap="twilight_shifted",vmin=-math.pi,vmax=math.pi)
        for ax in axes[r]: ax.set_xticks([]); ax.set_yticks([])
    add_frame(fig,"Matched SNR observations",5); save_page(fig,5,"matched_snr",paths)


def physical_page(ids: list[int], paths: list[Path]) -> None:
    keys=["dem","deformation_phase","coherence","phi_wrapped_clean","psi","region_id"]
    data=read_h5(128,"test_10dB",keys,ids)
    fig,axes=plt.subplots(5,5,figsize=(11.69,8.27),gridspec_kw={"top":.78,"bottom":.08,"left":.045,"right":.975,"hspace":.14,"wspace":.06})
    fig.suptitle("物理字段与 10 dB 观测链",x=.04,y=.905,ha="left",fontsize=19,weight="bold")
    fig.text(.04,.845,"从真实 DEM 到形变、相干度、干净包裹相位和含噪观测，展示数据生成链中的可审计字段。",fontsize=9.6,color=MUTED)
    titles=["DEM (m)","形变相位 (rad)","相干度","干净包裹相位","10 dB 观测"]
    for c,t in enumerate(titles): axes[0,c].set_title(t,fontsize=8.8,weight="bold")
    for r,idx in enumerate(ids):
        fields=[data["dem"][r],data["deformation_phase"][r],data["coherence"][r],data["phi_wrapped_clean"][r],data["psi"][r]]
        cmaps=["terrain","coolwarm","viridis","twilight_shifted","twilight_shifted"]
        for c,(img,cmap) in enumerate(zip(fields,cmaps)):
            kw={"vmin":-math.pi,"vmax":math.pi} if c>=3 else {}
            axes[r,c].imshow(img,cmap=cmap,**kw); axes[r,c].set_xticks([]); axes[r,c].set_yticks([])
        axes[r,0].set_ylabel(f"区域 {int(data['region_id'][r])}\nID {idx:04d}",fontsize=7.3)
    add_frame(fig,"Physical fields",6); save_page(fig,6,"physical_fields",paths)


def deformation_page(paths: list[Path]) -> list[int]:
    path=ROOT/"data"/"RBR128"/"train.h5"
    with h5py.File(path,"r") as f:
        codes=f["deformation_type"][:]
        ids=[int(np.flatnonzero(codes==code)[0]) for code in range(7)]
    data=read_h5(128,"train",["deformation_phase","phi","region_id"],ids)
    fig,axes=plt.subplots(2,7,figsize=(11.69,8.27),gridspec_kw={"top":.76,"bottom":.10,"left":.035,"right":.98,"hspace":.17,"wspace":.07})
    fig.suptitle("七类形变模式消融样本",x=.04,y=.905,ha="left",fontsize=19,weight="bold")
    fig.text(.04,.845,"上排为注入的形变相位，下排为地形与形变共同形成的展开真值；类型 0 不注入形变。",fontsize=9.6,color=MUTED)
    for c,(idx,name) in enumerate(zip(ids,DEFORMATION_NAMES)):
        vmax=max(float(np.max(np.abs(data["deformation_phase"][c]))),1e-5)
        axes[0,c].imshow(data["deformation_phase"][c],cmap="coolwarm",vmin=-vmax,vmax=vmax)
        axes[1,c].imshow(data["phi"][c],cmap="turbo")
        axes[0,c].set_title(f"{c} {name}\nID {idx}",fontsize=7.5)
        for r in range(2): axes[r,c].set_xticks([]); axes[r,c].set_yticks([])
    axes[0,0].set_ylabel("形变",fontsize=8); axes[1,0].set_ylabel("总相位",fontsize=8)
    add_frame(fig,"Deformation modes",7); save_page(fig,7,"deformation_modes",paths); return ids


def resolution_page(ids: list[int], paths: list[Path]) -> None:
    values={size:read_h5(size,"test_10dB",["phi","psi","scene_id","region_id"],ids) for size in (128,64,32)}
    fig,axes=plt.subplots(3,6,figsize=(11.69,8.27),gridspec_kw={"top":.77,"bottom":.09,"left":.05,"right":.97,"hspace":.20,"wspace":.08})
    fig.suptitle("跨分辨率场景一致性",x=.04,y=.905,ha="left",fontsize=19,weight="bold")
    fig.text(.04,.845,"同一行保持 scene_id、物理参数与地理来源一致；左三列为真值，右三列为 10 dB 包裹观测。",fontsize=9.6,color=MUTED)
    titles=["φ 128","φ 64","φ 32","ψ 128","ψ 64","ψ 32"]
    for c,t in enumerate(titles): axes[0,c].set_title(t,fontsize=9,weight="bold")
    for r,idx in enumerate(ids):
        for c,size in enumerate((128,64,32)):
            axes[r,c].imshow(values[size]["phi"][r],cmap="turbo")
            axes[r,c+3].imshow(values[size]["psi"][r],cmap="twilight_shifted",vmin=-math.pi,vmax=math.pi)
        sid=int(values[128]["scene_id"][r]); rid=int(values[128]["region_id"][r])
        axes[r,0].set_ylabel(f"区域 {rid}\nscene {sid}",fontsize=7.5)
        for ax in axes[r]: ax.set_xticks([]); ax.set_yticks([])
    add_frame(fig,"Cross-resolution alignment",8); save_page(fig,8,"cross_resolution",paths)


def statistics_page(stats: dict, paths: list[Path]) -> None:
    image=plt.imread(ROOT/"data"/"RBR_statistics.png")
    fig=plt.figure(figsize=(11.69,8.27)); fig.suptitle("数据统计与完整性审计",x=.04,y=.905,ha="left",fontsize=19,weight="bold")
    fig.text(.04,.845,"统计覆盖正式 H5 文件；审计检查数值范围、包裹关系、噪声残差、跨 SNR 与跨尺度不变量。",fontsize=9.6,color=MUTED)
    ax=fig.add_axes([.045,.09,.68,.68]); ax.imshow(image); ax.axis("off")
    fax=fig.add_axes([.755,.15,.21,.62]); fax.axis("off")
    fax.text(0,1,"审计结论",fontsize=14,weight="bold",va="top")
    items=["18 / 18 文件通过","30,000 条样本记录","所有数组均为有限值","包裹关系误差 = 0","跨 SNR 元数据一致","跨尺度来源一致"]
    for i,item in enumerate(items):
        fax.text(.02,.88-i*.10,"●",fontsize=9,color="#32865a",weight="bold")
        fax.text(.15,.88-i*.10,item,fontsize=9.2)
    fax.text(0,.20,"训练集摘要",fontsize=12,weight="bold")
    fax.text(0,.13,f"相位跨度：{stats['phase_span']['mean']:.2f} ± {stats['phase_span']['std']:.2f} rad",fontsize=8.2)
    fax.text(0,.075,f"平均相干度：{stats['coherence_mean']['mean']:.3f}",fontsize=8.2)
    fax.text(0,.02,f"DEM 高差：{stats['dem_range']['mean']:.1f} ± {stats['dem_range']['std']:.1f} m",fontsize=8.2)
    add_frame(fig,"Statistics and audit",9); save_page(fig,9,"statistics",paths)


def assemble(paths: list[Path]) -> None:
    width,height=landscape(A4); doc=canvas.Canvas(str(PDF),pagesize=(width,height),pageCompression=1)
    doc.setTitle("RBR Dataset Atlas"); doc.setAuthor("DMForPU")
    for path in paths:
        doc.drawImage(str(path),0,0,width=width,height=height,preserveAspectRatio=True,anchor="c")
        doc.showPage()
    doc.save()


def main() -> None:
    setup_style(); OUT.mkdir(parents=True,exist_ok=True); PAGES.mkdir(parents=True,exist_ok=True)
    for old in PAGES.glob("*.png"): old.unlink()
    config=yaml.safe_load((ROOT/"configs"/"rbr"/"regions.yaml").read_text(encoding="utf-8"))
    stats=json.loads((ROOT/"data"/"RBR_STATISTICS.json").read_text(encoding="utf-8"))["train_distribution_summary"]
    vector=shapefile.Reader(str(REGION_DIR/"rbr_spatial_splits.shp"),encoding="utf-8")
    assert len(vector)==13
    spans=phase_spans(); clean_ids=representatives(spans,12); region_ids=ids_by_region()
    resolution_ids=[region_ids[0],region_ids[2],region_ids[4]]
    selected={"clean":clean_ids,"regions":region_ids,"resolution":resolution_ids}
    pages=[]
    cover_page(stats,selected,pages); region_page(config["regions"],pages); dem_page(config["regions"],pages)
    clean_page(clean_ids,pages); snr_page(region_ids,pages); physical_page(region_ids,pages)
    selected["deformation"]=deformation_page(pages); resolution_page(resolution_ids,pages); statistics_page(stats,pages)
    assemble(pages)
    metadata={"pdf":str(PDF.relative_to(ROOT)),"pages":[str(p.relative_to(ROOT)) for p in pages],"selected_ids":selected,"region_vectors":str(REGION_DIR.relative_to(ROOT)),"page_count":len(pages)}
    META.write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(metadata,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
