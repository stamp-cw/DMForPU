"""Plot convergence diagnostics from the existing HF-epsilon history."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "experiments/results/gfs128_t200_prediction_study/runs/hf_epsilon"
OUT = RUN / "convergence_diagnostics"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(RUN / "history.csv")
valid = df[df["validated"] == True].copy()  # noqa: E712
best_i = valid["val_u3_aligned_nrmse"].idxmin()
best = valid.loc[best_i]
tail_n = min(100, len(df))
tail = df.tail(tail_n)
x = tail["epoch"].to_numpy(float)
coef = np.polyfit(x, tail["train_loss"].to_numpy(float), 1)[0]
first_loss = float(tail["train_loss"].iloc[0])
last_loss = float(tail["train_loss"].iloc[-1])
summary = {
    "epochs_available": int(df.epoch.max()),
    "best_u3_nrmse_epoch": int(best.epoch),
    "best_u3_nrmse": float(best.val_u3_aligned_nrmse),
    "best_u3_ssim_epoch": int(valid.loc[valid.val_u3_aligned_ssim.idxmax(), "epoch"]),
    "best_u3_ssim": float(valid.val_u3_aligned_ssim.max()),
    "last_epoch": int(df.epoch.iloc[-1]),
    "last_train_loss": float(df.train_loss.iloc[-1]),
    "last_anchor_u3_nrmse": float(valid.iloc[-1].val_u3_aligned_nrmse),
    "last_100_loss_ratio": last_loss / first_loss,
    "last_100_loss_linear_slope": float(coef),
    "validated_epochs": [int(v) for v in valid.epoch.tolist()],
}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

fig, axes = plt.subplots(2, 3, figsize=(17, 9))
ax = axes[0, 0]
ax.plot(df.epoch, df.train_loss, color="#3465a4", lw=1.4)
ax.set_yscale("log"); ax.set_title("Training loss (log scale)"); ax.set_xlabel("Epoch"); ax.set_ylabel("MSE noise objective")
ax = axes[0, 1]
ax.plot(valid.epoch, valid.val_u3_aligned_nrmse, "o-", label="U3 NRMSE")
ax.plot(valid.epoch, valid.val_mean_aligned_nrmse, "s--", label="mean NRMSE")
ax.axvline(best.epoch, color="r", ls="--", alpha=.7, label=f"best U3 {int(best.epoch)}")
ax.set_title("Validation NRMSE (25-step anchor)"); ax.set_xlabel("Epoch"); ax.legend(fontsize=8)
ax = axes[0, 2]
ax.plot(valid.epoch, valid.val_u3_aligned_ssim, "o-", color="#4e9a06")
ax.set_title("Validation U3-aligned SSIM"); ax.set_xlabel("Epoch"); ax.set_ylabel("SSIM")
ax = axes[1, 0]
ax.plot(valid.epoch, valid.val_u3_aligned_mae, "o-", label="MAE")
ax.plot(valid.epoch, valid.val_u3_aligned_rmse, "s--", label="RMSE")
ax.set_title("Validation U3-aligned errors"); ax.set_xlabel("Epoch"); ax.legend(fontsize=8)
ax = axes[1, 1]
ax.plot(df.epoch, df.lr, color="#75507b")
ax.set_title("Learning rate"); ax.set_xlabel("Epoch"); ax.set_ylabel("lr")
ax = axes[1, 2]
ax.plot(df.epoch, df.seconds, color="#c17d11", label="epoch time")
ax2 = ax.twinx(); ax2.plot(df.epoch, df.samples_per_second, color="#cc0000", alpha=.75, label="throughput")
ax.set_title("Training efficiency"); ax.set_xlabel("Epoch"); ax.set_ylabel("seconds")
ax2.set_ylabel("samples/s")
for a in axes.flat: a.grid(alpha=.25)
fig.suptitle("HF epsilon t200 convergence diagnostics", fontsize=15)
fig.tight_layout(); fig.savefig(OUT / "hf_epsilon_t200_convergence.png", dpi=220); plt.close(fig)
print(json.dumps(summary, indent=2, ensure_ascii=False))
