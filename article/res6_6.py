import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 8,
    "axes.labelsize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    # latex
    # "text.usetex": True,
    # "font.family": "serif"
})

categories = ["PUNet", "U3Net", "DLPU", "SQD-LSTM", "Restormer", "Uformer", "Ours"]
x = np.arange(len(categories))

nrmse = {
    "screp+wwf": [0.0492, 0.046, 0.0254, 0.0281, 0.028, 0.0301, np.nan],   # 没有 Apple
    "screp":  [0.0492+0.02, 0.046+0.02, 0.0254+0.02, 0.0281+0.02, 0.028+0.02, 0.0301+0.02, np.nan],   # 没有 Orange
    "wwf":  [0.0492+0.01, 0.046+0.01, 0.0254+0.01, 0.0281+0.01, 0.028+0.01, 0.0301+0.01, np.nan],
}

# plt.figure(figsize=(4.5, 3))
#
# for market, price in prices.items():
#     plt.plot(x, price, marker="o", label=market)
#
# plt.xticks(x, categories)
# plt.xlabel("Product Category")
# plt.ylabel("Price ($)")
# plt.legend(frameon=False)
#
# plt.grid(True, linestyle="--", alpha=0.6)
#
# plt.tight_layout()
# plt.show()

fig_dpi = 600
pdf_img_path = r"res/res6/figure.pdf"
png_img_path = r"res/res6/figure.png"


plt.figure(figsize=(4.5, 3))

for market, price in nrmse.items():
    price = np.array(price)
    mask = ~np.isnan(price)

    plt.scatter(
        x[mask],
        price[mask],
        label=market,
        s=50
    )

plt.xticks(x, categories)
plt.xlabel("Method")
plt.ylabel("NRMSE")
plt.legend(frameon=False)
plt.grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
plt.savefig(png_img_path, dpi=fig_dpi, bbox_inches="tight", pad_inches=0)
plt.savefig(pdf_img_path, dpi=fig_dpi, bbox_inches="tight", pad_inches=0)
plt.show()
