"""Plot the currently available HF-epsilon training/validation history."""
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "experiments/results/gfs128_t200_prediction_study/runs/hf_epsilon"
OUT = RUN / "current_progress"
OUT.mkdir(parents=True, exist_ok=True)
d = pd.read_csv(RUN / "history.csv")
valid = d[d["validated"] == True]
best = valid.loc[valid["val_u3_aligned_nrmse"].idxmin()]
fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
ax[0].plot(d["epoch"], d["train_loss"], color="#3465a4")
ax[0].set_yscale("log")
ax[0].set_title("HF epsilon training loss")
ax[0].set_xlabel("Epoch")
ax[0].set_ylabel("MSE(noise)")
ax[1].plot(valid["epoch"], valid["val_u3_aligned_nrmse"], "o-", label="U3-NRMSE", color="#75507b")
ax[1].plot(valid["epoch"], valid["val_u3_aligned_ssim"], "s-", label="SSIM", color="#4e9a06")
ax[1].axvline(best["epoch"], linestyle="--", color="#cc0000", label=f"best NRMSE: epoch {int(best['epoch'])}")
ax[1].set_title("25-step validation history")
ax[1].set_xlabel("Epoch")
ax[1].legend(fontsize=8)
for axis in ax:
    axis.grid(alpha=.25)
fig.tight_layout()
fig.savefig(OUT / "hf_epsilon_training_progress.png", dpi=220)
plt.close(fig)
(OUT / "summary.json").write_text(json.dumps({"available_through_epoch": int(d.epoch.max()),
    "best_validated_epoch": int(best.epoch), "best_val_u3_aligned_nrmse": float(best.val_u3_aligned_nrmse),
    "best_val_u3_aligned_ssim": float(best.val_u3_aligned_ssim), "inference_steps": 25},
    indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps({"available_through_epoch": int(d.epoch.max()), "best_epoch": int(best.epoch),
                  "best_u3_nrmse": float(best.val_u3_aligned_nrmse)}, ensure_ascii=False))
