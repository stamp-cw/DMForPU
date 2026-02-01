import seaborn as sns
import matplotlib.pyplot as plt


import pandas as pd

df = pd.DataFrame({
    "EncoderRMSE": [0.005, 0.01, 0.02,
                    0.005, 0.01, 0.02,
                    0.005, 0.01, 0.02],
    "ExtendedRMSE": [0.85, 0.95, 1.15,      # RWA
                     0.08, 0.10, 0.12,      # PCA
                     0.09, 0.11, 0.13],     # RWA+PCA
    "Method": ["RWA"] * 3 + ["PCA"] * 3 + ["RWA+PCA"] * 3,
    "CompactBands": ["3", "20", "20"] * 3
})


sns.lineplot(
    data=df,
    x="EncoderRMSE",
    y="ExtendedRMSE",
    hue="Method",
    style="CompactBands",
    markers=True,
    dashes=True
)
plt.grid(True, linestyle="--", alpha=0.6)
plt.xlabel("Encoder-Decoder RMSE")
plt.ylabel("Extended RMSE")
plt.title("Extended RMSE Trend Lines ↓")
plt.tight_layout()
plt.show()
