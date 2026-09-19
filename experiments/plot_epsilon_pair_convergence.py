"""Joint convergence plots for HF-epsilon and directional-epsilon t200."""
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments/results/gfs128_t200_prediction_study/runs"
OUT = BASE.parent / "epsilon_pair_convergence"
OUT.mkdir(parents=True, exist_ok=True)
paths = {"HF-epsilon": BASE/"hf_epsilon/history.csv", "Direction-epsilon": BASE/"directional_epsilon/history.csv"}
dfs = {k: pd.read_csv(v).query("epoch <= 600") for k,v in paths.items()}
valid = {k: d[d.validated == True].copy() for k,d in dfs.items()}  # noqa: E712

summary = {}
for name, d in valid.items():
    best = d.loc[d.val_u3_aligned_nrmse.idxmin()]
    best_ssim = d.loc[d.val_u3_aligned_ssim.idxmax()]
    best_mean = d.loc[d.val_mean_aligned_nrmse.idxmin()]
    summary[name] = {
        "best_u3_nrmse_epoch": int(best.epoch), "best_u3_nrmse": float(best.val_u3_aligned_nrmse),
        "best_u3_ssim_epoch": int(best_ssim.epoch), "best_u3_ssim": float(best_ssim.val_u3_aligned_ssim),
        "best_mean_nrmse_epoch": int(best_mean.epoch), "best_mean_nrmse": float(best_mean.val_mean_aligned_nrmse),
        "last_u3_nrmse": float(d.iloc[-1].val_u3_aligned_nrmse), "last_u3_ssim": float(d.iloc[-1].val_u3_aligned_ssim),
        "last_mean_nrmse": float(d.iloc[-1].val_mean_aligned_nrmse),
    }
checkpoints = [160, 200, 300, 400, 500, 580, 600]
for name, d in valid.items():
    summary[name]["checkpoints"] = {}
    for e in checkpoints:
        row = d[d.epoch == e]
        if not row.empty:
            row = row.iloc[0]
            summary[name]["checkpoints"][str(e)] = {k: float(row[k]) for k in ("val_u3_aligned_nrmse", "val_u3_aligned_ssim", "val_mean_aligned_nrmse")}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

colors = {"HF-epsilon": "#3465a4", "Direction-epsilon": "#cc4c02"}
fig, ax = plt.subplots(2, 3, figsize=(17, 9))
metrics = [("train_loss", "Training loss (log)", True), ("val_u3_aligned_nrmse", "Validation U3-NRMSE", False),
           ("val_u3_aligned_ssim", "Validation U3-SSIM", False), ("val_mean_aligned_nrmse", "Validation mean NRMSE", False),
           ("val_u3_aligned_mae", "Validation U3-MAE", False), ("val_u3_aligned_rmse", "Validation U3-RMSE", False)]
for axis, (key, title, logy) in zip(ax.flat, metrics):
    for name, d in dfs.items():
        source = valid[name] if key.startswith("val_") else d
        axis.plot(source.epoch, source[key], "o-" if key.startswith("val_") else "-", ms=3, lw=1.5, label=name, color=colors[name])
    if logy: axis.set_yscale("log")
    axis.set_title(title); axis.set_xlabel("Epoch"); axis.grid(alpha=.25)
    if key == "val_u3_aligned_nrmse": axis.legend(fontsize=8)
fig.suptitle("GFS128 t200: HF-epsilon vs directional-epsilon convergence", fontsize=15)
fig.tight_layout(); fig.savefig(OUT / "hf_vs_direction_epsilon_convergence.png", dpi=220); plt.close(fig)

h = valid["HF-epsilon"].set_index("epoch")
r = valid["Direction-epsilon"].set_index("epoch")
common = h.index.intersection(r.index)
gap = pd.DataFrame({"epoch": common,
    "u3_nrmse_direction_minus_hf": r.loc[common, "val_u3_aligned_nrmse"].to_numpy() - h.loc[common, "val_u3_aligned_nrmse"].to_numpy(),
    "mean_nrmse_direction_minus_hf": r.loc[common, "val_mean_aligned_nrmse"].to_numpy() - h.loc[common, "val_mean_aligned_nrmse"].to_numpy()})
fig, axis = plt.subplots(figsize=(10, 4.5)); axis.axhline(0, color='k', lw=1)
axis.plot(gap.epoch, gap.u3_nrmse_direction_minus_hf, 'o-', label='U3-NRMSE gap')
axis.plot(gap.epoch, gap.mean_nrmse_direction_minus_hf, 's-', label='mean-NRMSE gap')
axis.set_title('Directional minus HF validation gap (negative = directional better)'); axis.set_xlabel('Epoch'); axis.set_ylabel('metric gap'); axis.grid(alpha=.25); axis.legend()
fig.tight_layout(); fig.savefig(OUT / "direction_minus_hf_gap.png", dpi=220); plt.close(fig)
gap.to_csv(OUT / "direction_minus_hf_gap.csv", index=False)
print(json.dumps(summary, indent=2, ensure_ascii=False))
