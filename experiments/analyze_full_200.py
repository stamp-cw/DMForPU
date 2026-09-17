"""Create tables, plots and a Chinese report for the full 200-epoch study."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "results" / "full_200"
METHODS = ("punet", "sqd_lstm", "u3net")
LABELS = {"punet": "PUNet", "sqd_lstm": "SQD-LSTM", "u3net": "U3Net"}


def load_history(method: str) -> list[dict[str, float | str]]:
    with (OUT / "runs" / method / "history.csv").open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    converted = []
    for row in rows:
        converted.append({key: value if key == "phase" else float(value) for key, value in row.items()})
    return converted


def main() -> None:
    histories = {method: load_history(method) for method in METHODS}
    results = {method: json.loads((OUT / "runs" / method / "complete.json").read_text(encoding="utf-8"))
               for method in METHODS}
    integrity = {}
    summary = []
    for method in METHODS:
        folder = OUT / "runs" / method
        history = histories[method]
        weights = list((folder / "weights").glob("epoch_*.pth"))
        integrity[method] = {
            "history_rows": len(history), "weight_files": len(weights),
            "epochs_with_failed_updates": sum(int(r["successful_updates"]) != int(r["batches"]) for r in history),
            "all_finite": all(np.isfinite(v) for r in history for k, v in r.items() if k != "phase"),
        }
        result, eff = results[method], results[method]["efficiency"]
        summary.append({
            "method": LABELS[method], "selected_epoch": result["selected_epoch"],
            "test_mae": result["test"]["mae"], "test_rmse": result["test"]["rmse"],
            "test_aligned_mae": result["test"]["aligned_mae"],
            "test_aligned_rmse": result["test"]["aligned_rmse"],
            "test_aligned_nrmse": result["test"]["aligned_nrmse"], "test_pge": result["test"]["pge"],
            "test_rewrap_mae": result["test"]["rewrap_circular_mae"],
            "parameters": eff["parameters"], "model_state_mib": eff["model_state_mib"],
            "latency_batch1_ms": eff["latency_batch1_ms_mean"], "flops_batch1": eff["flops_batch1"],
            "total_train_seconds": sum(r["train_seconds"] for r in history),
            "total_epoch_seconds": sum(r["epoch_seconds"] for r in history),
            "median_train_samples_per_second": float(np.median([r["train_samples_per_second"] for r in history])),
            "peak_training_allocated_mib": max(r["gpu_peak_allocated_mib"] for r in history),
            "final_train_loss": history[-1]["train_loss"], "final_val_aligned_mae": history[-1]["val_aligned_mae"],
        })
    if not all(x["history_rows"] == x["weight_files"] == 200 and not x["epochs_with_failed_updates"] and x["all_finite"]
               for x in integrity.values()):
        raise RuntimeError(f"Incomplete result: {integrity}")

    with (OUT / "comparison.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0])); writer.writeheader(); writer.writerows(summary)
    (OUT / "comparison.json").write_text(json.dumps({"integrity": integrity, "summary": summary}, indent=2,
                                                      ensure_ascii=False), encoding="utf-8")

    figures = OUT / "figures"; figures.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for method in METHODS:
        h = histories[method]; epoch = [r["epoch"] for r in h]
        axes[0, 0].plot(epoch, [r["train_loss"] for r in h], label=LABELS[method])
        axes[0, 1].plot(epoch, [r["val_aligned_mae"] for r in h], label=LABELS[method])
        axes[1, 0].plot(epoch, [r["train_samples_per_second"] for r in h], label=LABELS[method])
        axes[1, 1].plot(epoch, [r["epoch_seconds"] for r in h], label=LABELS[method])
    axes[0, 0].set(title="Training loss", xlabel="Epoch", ylabel="Objective"); axes[0, 0].set_yscale("log")
    axes[0, 1].set(title="Validation aligned MAE", xlabel="Epoch", ylabel="rad"); axes[0, 1].set_yscale("log")
    axes[1, 0].set(title="Training throughput", xlabel="Epoch", ylabel="samples/s")
    axes[1, 1].set(title="Epoch time (train + validation)", xlabel="Epoch", ylabel="s")
    for axis in axes.flat:
        axis.grid(alpha=.25); axis.legend()
    axes[0, 0].axvline(143.5, color="gray", linestyle="--", linewidth=1)
    axes[0, 1].axvline(143.5, color="gray", linestyle="--", linewidth=1, label="U3Net distillation")
    fig.savefig(figures / "training_curves.png", dpi=180); plt.close(fig)

    names = [row["method"] for row in summary]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    axes[0].bar(names, [row["test_aligned_mae"] for row in summary]); axes[0].set_ylabel("rad"); axes[0].set_title("Test aligned MAE")
    axes[1].bar(names, [row["latency_batch1_ms"] for row in summary]); axes[1].set_ylabel("ms/image"); axes[1].set_title("Batch-1 latency")
    axes[2].bar(names, [row["flops_batch1"] / 1e9 for row in summary]); axes[2].set_ylabel("GFLOPs/image"); axes[2].set_title("Compute")
    for axis in axes: axis.grid(axis="y", alpha=.25)
    fig.savefig(figures / "method_comparison.png", dpi=180); plt.close(fig)

    by = {row["method"]: row for row in summary}; p, s, u = by["PUNet"], by["SQD-LSTM"], by["U3Net"]
    report = f"""# PUNet、SQD-LSTM、U3Net 完整 200 轮对比

三种方法均在 SyntheticPUMat128Big 的全部 20,000 对训练数据上从头训练 200 轮。验证集为官方测试集固定拆出的 500 对；下表为各自验证集最佳权重在其余独立 1,500 对测试集上的结果。每种方法均有 200 个逐轮权重和 200 行有限指标，所有轮次均完成全部优化器更新。

| 方法 | 最佳轮次 | 对齐 MAE↓ | 对齐 RMSE↓ | NRMSE↓ | PGE↓ | 重缠绕 MAE↓ |
|---|---:|---:|---:|---:|---:|---:|
| PUNet | {p['selected_epoch']} | {p['test_aligned_mae']:.4f} | {p['test_aligned_rmse']:.4f} | {p['test_aligned_nrmse']:.4f} | {p['test_pge']:.4f} | {p['test_rewrap_mae']:.4f} |
| SQD-LSTM | {s['selected_epoch']} | {s['test_aligned_mae']:.4f} | {s['test_aligned_rmse']:.4f} | {s['test_aligned_nrmse']:.4f} | {s['test_pge']:.4f} | {s['test_rewrap_mae']:.4f} |
| U3Net | {u['selected_epoch']} | {u['test_aligned_mae']:.4f} | {u['test_aligned_rmse']:.4f} | {u['test_aligned_nrmse']:.4f} | {u['test_pge']:.4f} | {u['test_rewrap_mae']:.4f} |

PUNet 在全部精度指标上明显最好。其测试对齐 MAE 比 SQD-LSTM 低 {(1-p['test_aligned_mae']/s['test_aligned_mae'])*100:.1f}%，比 U3Net 低 {(1-p['test_aligned_mae']/u['test_aligned_mae'])*100:.1f}%。PUNet 的原始 MAE 与对齐 MAE相同，说明它也学到了绝对相位偏移。

SQD-LSTM 的原始 MAE 为 {s['test_mae']:.4f} rad，而对齐后为 {s['test_aligned_mae']:.4f} rad。这符合 `VAR + 0.1 TV` 损失的性质：对预测加任意全局常数不会改变损失，因此该方法无法从当前目标函数确定绝对偏移。即使消除全局 2π 偏移，其精度仍远低于 PUNet。

U3Net 最佳权重出现在第 {u['selected_epoch']} 轮，早于第 144 轮开始的教师蒸馏。第 200 轮验证对齐 MAE 为 {u['final_val_aligned_mae']:.4f} rad，显著差于最佳值 {results['u3net']['best_val_aligned_mae']:.4f} rad；说明本次缩短到 200 轮后，原始 143+57 比例下的蒸馏没有带来收益。正式比较应使用已保存的第 {u['selected_epoch']} 轮最佳权重。

| 方法 | 参数量 | 权重大小 | 单张延迟↓ | FLOPs↓ | 训练耗时 | 中位吞吐 |
|---|---:|---:|---:|---:|---:|---:|
| PUNet | {p['parameters']:,} | {p['model_state_mib']:.2f} MiB | {p['latency_batch1_ms']:.2f} ms | {p['flops_batch1']/1e9:.2f} G | {p['total_epoch_seconds']/3600:.2f} h | {p['median_train_samples_per_second']:.0f} 张/s |
| SQD-LSTM | {s['parameters']:,} | {s['model_state_mib']:.2f} MiB | {s['latency_batch1_ms']:.2f} ms | {s['flops_batch1']/1e9:.2f} G | {s['total_epoch_seconds']/3600:.2f} h | {s['median_train_samples_per_second']:.0f} 张/s |
| U3Net | {u['parameters']:,} | {u['model_state_mib']:.2f} MiB | {u['latency_batch1_ms']:.2f} ms | {u['flops_batch1']/1e9:.2f} G | {u['total_epoch_seconds']/3600:.2f} h | {u['median_train_samples_per_second']:.0f} 张/s |

SQD-LSTM 的单张延迟略低于 PUNet，但参数量约为 PUNet 的 {s['parameters']/p['parameters']:.2f} 倍、计算量约为 {s['flops_batch1']/p['flops_batch1']:.2f} 倍。U3Net 的单张推理延迟约为 PUNet 的 {u['latency_batch1_ms']/p['latency_batch1_ms']:.1f} 倍，计算量约为 {u['flops_batch1']/p['flops_batch1']:.1f} 倍，而且精度更低。综合本数据集上的精度、模型大小和计算量，PUNet 是三者中最好的选择。
"""
    (OUT / "REPORT.zh-CN.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
