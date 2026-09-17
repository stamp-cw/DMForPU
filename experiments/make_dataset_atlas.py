"""Build a reproducible visual atlas for the GFS128 and RME128 datasets."""
from __future__ import annotations

import json
import math
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
OUTPUT = ROOT / "output" / "pdf"
PAGES = OUTPUT / "dataset_atlas_gfs128_rme128_pages"
PDF = OUTPUT / "dataset_atlas_gfs128_rme128.pdf"
META = OUTPUT / "dataset_atlas_gfs128_rme128.json"
FAMILIES = ("GFS128", "RME128")
SNRS = (30, 20, 10, 5, 0)
SEED = 20240913
BG, INK, MUTED = "#f5f7fb", "#152238", "#5d697a"
COLORS = {"GFS128": "#1677b8", "RME128": "#d26432"}


def setup_style() -> None:
    font_path = Path(r"C:\Windows\Fonts\msyh.ttc")
    if font_path.exists():
        fm.fontManager.addfont(font_path)
        font_name = fm.FontProperties(fname=font_path).get_name()
    else:
        font_name = "DejaVu Sans"
    plt.rcParams.update(
        {
            "font.family": font_name,
            "axes.unicode_minus": False,
            "figure.facecolor": BG,
            "axes.facecolor": "white",
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "savefig.facecolor": BG,
        }
    )


def h5_path(family: str, split: str) -> Path:
    return ROOT / "data" / family / f"{split}.h5"


def read_indices(family: str, split: str, key: str, indices: list[int]) -> np.ndarray:
    # h5py requires increasing indices; restore the requested presentation order.
    order = np.argsort(indices)
    sorted_ids = np.asarray(indices)[order]
    with h5py.File(h5_path(family, split), "r") as f:
        values = f[key][sorted_ids]
    inverse = np.argsort(order)
    return values[inverse]


def phase_spans(family: str) -> np.ndarray:
    with h5py.File(h5_path(family, "test_30dB"), "r") as f:
        phi = f["phi"]
        spans = np.empty(len(phi), np.float32)
        for begin in range(0, len(phi), 100):
            x = phi[begin : begin + 100]
            spans[begin : begin + len(x)] = np.ptp(x, axis=(1, 2))
    return spans


def representative_ids(spans: np.ndarray, count: int) -> list[int]:
    ranked = np.argsort(spans)
    positions = np.linspace(0.03, 0.97, count)
    return [int(ranked[round(q * (len(ranked) - 1))]) for q in positions]


def train_ids(family: str, count: int = 8) -> list[int]:
    with h5py.File(h5_path(family, "train"), "r") as f:
        snr = f["snr"][:]
    rng = np.random.default_rng(SEED + (0 if family == "GFS128" else 1))
    chosen: list[int] = []
    target = [0, 5, 10, 20, 30, 60, 0, 10]
    for level in target[:count]:
        candidates = np.flatnonzero(np.isclose(snr, level))
        chosen.append(int(rng.choice(candidates)))
    return chosen


def add_page_frame(fig: plt.Figure, section: str, page: int) -> None:
    fig.text(0.035, 0.965, "DMForPU · 数据集图册", fontsize=9.5, weight="bold", color=MUTED)
    fig.text(0.965, 0.965, section, fontsize=9.5, ha="right", color=MUTED)
    fig.add_artist(plt.Line2D([0.035, 0.965], [0.948, 0.948], transform=fig.transFigure, color="#d8dee8", lw=0.8))
    fig.text(0.035, 0.022, "固定索引与色标；生成脚本：experiments/make_dataset_atlas.py", fontsize=7.5, color=MUTED)
    fig.text(0.965, 0.022, f"{page:02d}", fontsize=8, ha="right", color=MUTED)


def save_page(fig: plt.Figure, number: int, slug: str, paths: list[Path]) -> None:
    path = PAGES / f"{number:02d}_{slug}.png"
    fig.savefig(path, dpi=180, bbox_inches=None, pad_inches=0)
    plt.close(fig)
    paths.append(path)


def clean_gallery(family: str, ids: list[int], page: int, paths: list[Path]) -> None:
    phases = read_indices(family, "test_30dB", "phi", ids)
    fig, axes = plt.subplots(3, 4, figsize=(11.69, 8.27), gridspec_kw={"top": 0.81, "bottom": 0.085, "hspace": 0.30, "wspace": 0.12})
    color = COLORS[family]
    fig.suptitle(f"{family} · 干净展开相位形态", x=0.04, y=0.915, ha="left", fontsize=20, weight="bold", color=color)
    fig.text(0.04, 0.878, "按相位跨度分位数选取 12 个样本；每幅图独立色标范围，以突出空间结构。", fontsize=9.5, color=MUTED)
    for ax, sample, idx in zip(axes.flat, phases, ids):
        lo, hi = float(sample.min()), float(sample.max())
        ax.imshow(sample, cmap="turbo", vmin=lo, vmax=hi, interpolation="nearest")
        ax.set_title(f"ID {idx:04d}   Δφ={hi-lo:.1f} rad", fontsize=8.5, pad=4)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values(): spine.set_color("#d6dde8")
    add_page_frame(fig, "Clean phase gallery", page)
    save_page(fig, page, f"{family.lower()}_clean_gallery", paths)


def snr_gallery(family: str, ids: list[int], page: int, paths: list[Path]) -> None:
    truth = read_indices(family, "test_30dB", "phi", ids)
    noisy = {s: read_indices(family, f"test_{s}dB", "psi", ids) for s in SNRS}
    fig, axes = plt.subplots(4, 6, figsize=(11.69, 8.27), gridspec_kw={"top": 0.81, "bottom": 0.10, "left": 0.05, "right": 0.965, "hspace": 0.18, "wspace": 0.08})
    fig.suptitle(f"{family} · 同一场景在不同噪声等级下的观测", x=0.04, y=0.915, ha="left", fontsize=19, weight="bold", color=COLORS[family])
    fig.text(0.04, 0.875, "第一列为展开真值 φ；其余各列为含噪包裹相位 ψ，统一使用 [-π, π] 色标。", fontsize=9.5, color=MUTED)
    headers = ["真值 φ"] + [f"ψ · {s} dB" for s in SNRS]
    for col, title in enumerate(headers): axes[0, col].set_title(title, fontsize=9.5, weight="bold", pad=5)
    for row, (idx, clean) in enumerate(zip(ids, truth)):
        axes[row, 0].imshow(clean, cmap="turbo", interpolation="nearest")
        axes[row, 0].set_ylabel(f"ID {idx:04d}", fontsize=8.5, rotation=90, labelpad=5)
        for col, snr in enumerate(SNRS, 1):
            axes[row, col].imshow(noisy[snr][row], cmap="twilight_shifted", vmin=-math.pi, vmax=math.pi, interpolation="nearest")
        for ax in axes[row]:
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values(): spine.set_color("#d6dde8")
    add_page_frame(fig, "Matched SNR comparison", page)
    save_page(fig, page, f"{family.lower()}_snr_gallery", paths)


def training_gallery(family: str, ids: list[int], page: int, paths: list[Path]) -> None:
    clean = read_indices(family, "train", "phi", ids)
    wrapped = read_indices(family, "train", "psi", ids)
    snr = read_indices(family, "train", "snr", ids)
    fig, axes = plt.subplots(4, 4, figsize=(11.69, 8.27), gridspec_kw={"top": 0.81, "bottom": 0.09, "left": 0.055, "right": 0.965, "hspace": 0.28, "wspace": 0.10})
    fig.suptitle(f"{family} · 训练集样本对", x=0.04, y=0.915, ha="left", fontsize=19, weight="bold", color=COLORS[family])
    fig.text(0.04, 0.875, "每组左侧为展开真值，右侧为训练时输入的含噪包裹相位；覆盖训练集全部 SNR 档位。", fontsize=9.5, color=MUTED)
    for k, (idx, gt, obs, level) in enumerate(zip(ids, clean, wrapped, snr)):
        row, pair = divmod(k, 2); left = pair * 2
        axes[row, left].imshow(gt, cmap="turbo", interpolation="nearest")
        axes[row, left + 1].imshow(obs, cmap="twilight_shifted", vmin=-math.pi, vmax=math.pi, interpolation="nearest")
        axes[row, left].set_title(f"ID {idx:04d} · 真值", fontsize=8.5)
        axes[row, left + 1].set_title(f"输入 · {float(level):g} dB", fontsize=8.5)
    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values(): spine.set_color("#d6dde8")
    add_page_frame(fig, "Training pairs", page)
    save_page(fig, page, f"{family.lower()}_training_pairs", paths)


def family_stats(family: str) -> dict[str, np.ndarray | float]:
    spans, stds, gradients = [], [], []
    with h5py.File(h5_path(family, "test_30dB"), "r") as f:
        phi = f["phi"]
        for begin in range(0, len(phi), 100):
            x = phi[begin : begin + 100].astype(np.float64)
            spans.append(np.ptp(x, axis=(1, 2)))
            stds.append(np.std(x, axis=(1, 2)))
            gx, gy = np.diff(x, axis=2), np.diff(x, axis=1)
            gradients.append(np.sqrt((np.mean(gx * gx, axis=(1, 2)) + np.mean(gy * gy, axis=(1, 2))) / 2))
    circular = []
    for snr in SNRS:
        total = 0.0; count = 0
        with h5py.File(h5_path(family, f"test_{snr}dB"), "r") as f:
            for begin in range(0, len(f["phi"]), 100):
                phi = f["phi"][begin : begin + 100]
                psi = f["psi"][begin : begin + 100]
                clean_wrapped = np.remainder(phi + math.pi, 2 * math.pi) - math.pi
                delta = np.remainder(psi - clean_wrapped + math.pi, 2 * math.pi) - math.pi
                total += float(np.abs(delta).sum()); count += delta.size
        circular.append(total / count)
    with h5py.File(h5_path(family, "train"), "r") as f: train_snr = f["snr"][:]
    return {"span": np.concatenate(spans), "std": np.concatenate(stds), "gradient": np.concatenate(gradients), "circular_mae": np.asarray(circular), "train_snr": train_snr}


def statistics_page(stats: dict[str, dict], page: int, paths: list[Path]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.69, 8.27), gridspec_kw={"top": 0.84, "bottom": 0.11, "left": 0.08, "right": 0.96, "hspace": 0.34, "wspace": 0.27})
    fig.suptitle("GFS128 与 RME128 · 数据分布概览", x=0.04, y=0.915, ha="left", fontsize=19, weight="bold")
    fig.text(0.04, 0.875, "统计基于 1,000 个固定测试场景；噪声误差按圆周距离计算。", fontsize=9.5, color=MUTED)
    plots = (("span", "单幅相位跨度 Δφ (rad)"), ("std", "单幅相位标准差 (rad)"), ("gradient", "空间梯度 RMS (rad/pixel)"))
    for ax, (key, label) in zip(axes.flat[:3], plots):
        for family in FAMILIES:
            ax.hist(stats[family][key], bins=28, density=True, alpha=0.48, color=COLORS[family], label=family)
        ax.set_xlabel(label); ax.set_ylabel("概率密度"); ax.grid(alpha=0.18); ax.legend(frameon=False, fontsize=8.5)
    ax = axes[1, 1]
    for family in FAMILIES:
        ax.plot(SNRS, stats[family]["circular_mae"], marker="o", lw=2, color=COLORS[family], label=family)
    ax.set_xlabel("SNR (dB)"); ax.set_ylabel("包裹观测圆周 MAE (rad)"); ax.set_xticks(SNRS); ax.invert_xaxis(); ax.grid(alpha=0.22); ax.legend(frameon=False, fontsize=8.5)
    add_page_frame(fig, "Dataset statistics", page)
    save_page(fig, page, "dataset_statistics", paths)


def cover_page(manifest: dict, selected: dict, page: int, paths: list[Path]) -> None:
    fig = plt.figure(figsize=(11.69, 8.27))
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.add_patch(plt.Rectangle((0, 0.62), 1, 0.38, color=INK, transform=ax.transAxes))
    fig.text(0.065, 0.855, "GFS128 · RME128", fontsize=35, weight="bold", color="white")
    fig.text(0.065, 0.765, "相位解缠数据集图册", fontsize=25, weight="bold", color="white")
    fig.text(0.067, 0.695, "Dataset Atlas for Phase Unwrapping", fontsize=13, color="#c9d8ea")
    fig.text(0.065, 0.55, "数据协议", fontsize=15, weight="bold", color=INK)
    rows = [
        ("分辨率", "128 × 128"),
        ("训练集", "每类 5,000 张；SNR ∈ {0, 5, 10, 20, 30, 60} dB"),
        ("测试集", "每类每个 SNR 1,000 张；0/5/10/20/30 dB，共 5,000 张"),
        ("字段", "φ：干净展开相位；ψ：含噪包裹相位；scene_id：场景编号"),
        ("生成随机种子", str(manifest["seed"])),
    ]
    y = 0.495
    for label, value in rows:
        fig.text(0.07, y, label, fontsize=10, weight="bold", color=MUTED)
        fig.text(0.20, y, value, fontsize=10.5, color=INK)
        y -= 0.055
    fig.text(0.065, 0.175, "代表样本", fontsize=15, weight="bold", color=INK)
    fig.text(0.07, 0.125, f"GFS128: {selected['GFS128']['representative']}", fontsize=9.5, color=COLORS["GFS128"])
    fig.text(0.07, 0.085, f"RME128: {selected['RME128']['representative']}", fontsize=9.5, color=COLORS["RME128"])
    fig.text(0.93, 0.045, datetime.now().strftime("%Y-%m-%d"), fontsize=9, ha="right", color=MUTED)
    save_page(fig, page, "cover", paths)


def assemble_pdf(page_paths: list[Path]) -> None:
    width, height = landscape(A4)
    c = canvas.Canvas(str(PDF), pagesize=(width, height), pageCompression=1)
    c.setTitle("GFS128 and RME128 Dataset Atlas")
    c.setAuthor("DMForPU")
    for path in page_paths:
        c.drawImage(str(path), 0, 0, width=width, height=height, preserveAspectRatio=True, anchor="c")
        c.showPage()
    c.save()


def main() -> None:
    setup_style(); OUTPUT.mkdir(parents=True, exist_ok=True); PAGES.mkdir(parents=True, exist_ok=True)
    for old in PAGES.glob("*.png"): old.unlink()
    manifest = json.loads((ROOT / "data" / "DATASETS_V2.json").read_text(encoding="utf-8"))
    spans = {family: phase_spans(family) for family in FAMILIES}
    selected = {
        family: {
            "representative": representative_ids(spans[family], 12),
            "snr_comparison": representative_ids(spans[family], 4),
            "training": train_ids(family),
        }
        for family in FAMILIES
    }
    pages: list[Path] = []
    cover_page(manifest, selected, 1, pages)
    clean_gallery("GFS128", selected["GFS128"]["representative"], 2, pages)
    clean_gallery("RME128", selected["RME128"]["representative"], 3, pages)
    snr_gallery("GFS128", selected["GFS128"]["snr_comparison"], 4, pages)
    snr_gallery("RME128", selected["RME128"]["snr_comparison"], 5, pages)
    training_gallery("GFS128", selected["GFS128"]["training"], 6, pages)
    training_gallery("RME128", selected["RME128"]["training"], 7, pages)
    stats = {family: family_stats(family) for family in FAMILIES}
    statistics_page(stats, 8, pages)
    assemble_pdf(pages)
    metadata = {
        "title": "GFS128 and RME128 Dataset Atlas",
        "created": datetime.now().isoformat(timespec="seconds"),
        "generator": str(Path(__file__).relative_to(ROOT)),
        "pdf": str(PDF.relative_to(ROOT)),
        "pages": [str(p.relative_to(ROOT)) for p in pages],
        "selection_seed": SEED,
        "selected_indices": selected,
        "statistics": {
            family: {
                "phase_span_mean": float(np.mean(stats[family]["span"])),
                "phase_span_std": float(np.std(stats[family]["span"])),
                "phase_std_mean": float(np.mean(stats[family]["std"])),
                "gradient_rms_mean": float(np.mean(stats[family]["gradient"])),
                "circular_mae_by_snr": {str(s): float(v) for s, v in zip(SNRS, stats[family]["circular_mae"])},
                "train_snr_counts": {str(int(s)): int(np.count_nonzero(np.isclose(stats[family]["train_snr"], s))) for s in (0, 5, 10, 20, 30, 60)},
            }
            for family in FAMILIES
        },
    }
    META.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(PDF)
    print(META)


if __name__ == "__main__":
    main()
