import hashlib
from pathlib import Path

import torch

import experiments.train_gfs128_upstream as study


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SOURCE_SHA256 = "24a676522f8e1edc1a03c9f0fc335f6311650d3afbb2a0bfcd0da196f6ab83b3"


def test_vurnet_upstream_source_and_protocol():
    source = ROOT / "third_party" / "VUR-Net" / "VURNet.py"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == EXPECTED_SOURCE_SHA256
    assert study.COMMITS["vurnet"] == "59168a529fe879daeac8fa2f03ea60b91ea4e0b3"
    assert study.EPOCHS["vurnet"] == 500
    assert study.BATCH["vurnet"] == 20


def test_vurnet_original_model_accepts_128():
    model = study.build("vurnet").eval()
    with torch.no_grad():
        output = study.forward_model("vurnet", model, torch.zeros(1, 1, 128, 128, device="cuda"))
    assert output.shape == (1, 1, 128, 128)
    assert sum(parameter.numel() for parameter in model.parameters()) == 21_561_430
