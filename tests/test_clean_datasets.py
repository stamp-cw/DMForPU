import unittest
from pathlib import Path

import h5py
import numpy as np

from dataset.U3SyntheticH5 import U3SyntheticH5


ROOT = Path(__file__).resolve().parents[1]


class CleanDatasetTests(unittest.TestCase):
    def test_loader_selects_clean_condition(self):
        dataset = U3SyntheticH5(ROOT / "data" / "GFS32", split="test", test_snr="clean")
        self.assertEqual(dataset.path.name, "test_clean.h5")
        self.assertTrue(np.isposinf(dataset[0]["snr"].numpy()).all())

    def test_all_clean_files_are_exact_and_paired(self):
        for family in ("GFS", "RME", "RTS"):
            for size in (128, 64, 32):
                folder = ROOT / "data" / f"{family}{size}"
                with h5py.File(folder / "test_clean.h5", "r") as clean, h5py.File(folder / "test_30dB.h5", "r") as paired:
                    self.assertTrue(np.isposinf(clean["snr"][:]).all())
                    self.assertTrue(np.array_equal(clean["phi"][:], paired["phi"][:]))
                    self.assertTrue(np.array_equal(clean["scene_id"][:], paired["scene_id"][:]))
                    for begin in range(0, len(clean["phi"]), 64):
                        end = min(len(clean["phi"]), begin + 64)
                        expected = (clean["phi_wrapped_clean"][begin:end] if family == "RTS" else
                                    np.angle(np.exp(1j * clean["phi"][begin:end])).astype(np.float32))
                        self.assertTrue(np.array_equal(clean["psi"][begin:end], expected))
                    if family == "RTS":
                        self.assertEqual(np.count_nonzero(clean["noise_map"][:]), 0)


if __name__ == "__main__":
    unittest.main()
