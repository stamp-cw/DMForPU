import matplotlib.pyplot as plt
import numpy as np

categories = ["Apple", "Banana", "Orange", "Milk"]
x = np.arange(len(categories))

# prices = {
#     "Walmart": [3.2, 2.1, 2.8, 4.5],
#     "Costco":  [3.0, 1.9, 2.6, 4.2],
#     "Target":  [3.4, 2.3, 2.9, 4.8],
# }

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
# # ✅ 打开网格（重点）
# plt.grid(True, linestyle="--", alpha=0.6)
#
# plt.tight_layout()
# plt.show()

prices = {
    "Walmart": [np.nan, 2.1, 2.8, 4.5],   # 没有 Apple
    "Costco":  [3.0, 1.9, np.nan, 4.2],   # 没有 Orange
    "Target":  [3.4, 2.3, 2.9, 4.8],
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

plt.figure(figsize=(4.5, 3))

for market, price in prices.items():
    price = np.array(price)
    mask = ~np.isnan(price)

    plt.scatter(
        x[mask],
        price[mask],
        label=market,
        s=50
    )

plt.xticks(x, categories)
plt.xlabel("Product Category")
plt.ylabel("Price ($)")
plt.legend(frameon=False)
plt.grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
plt.show()
