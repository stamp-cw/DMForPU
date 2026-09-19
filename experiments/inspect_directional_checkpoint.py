import torch
from pathlib import Path
for n in ("hf_epsilon_loss1", "directional_epsilon_loss1"):
    p = Path("experiments/results/gfs128_t200_prediction_study/runs") / n / "weights/epoch_400.pth"
    s = torch.load(p, map_location="cpu", weights_only=False)["model"]
    print(n, "params", sum(v.numel() for v in s.values()), "keys", len(s))
    for k, v in s.items():
        if any(q in k for q in ("gamma_logit", "alpha_logit", "temperature_logit", "distance_logit")):
            print(k, v.flatten()[:4].tolist(), "sigmoid", float(torch.sigmoid(v.flatten()[0])))
    print("cross keys", sum(1 for k in s if ".cross." in k))
