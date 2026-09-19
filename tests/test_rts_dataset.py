import math
import unittest

import numpy as np

from experiments.generate_rts_dataset import deformation_phase, tile_product, wrap_phase


class RTSGeneratorTests(unittest.TestCase):
    def test_wrap_range_and_periodicity(self):
        value=np.linspace(-100,100,10001,dtype=np.float32)
        wrapped=wrap_phase(value)
        self.assertGreaterEqual(float(wrapped.min()),-math.pi)
        self.assertLess(float(wrapped.max()),math.pi)
        self.assertLess(float(np.max(np.abs(np.angle(np.exp(1j*(wrapped-value)))))),2e-5)

    def test_deformation_is_finite_and_wavelength_consistent(self):
        rng=np.random.default_rng(5);wavelength=.055465
        phase,peak=deformation_phase("elliptical_bowl",64,wavelength,rng,[.25,4.0],False)
        self.assertEqual(phase.shape,(64,64));self.assertTrue(np.isfinite(phase).all())
        self.assertAlmostEqual(peak,float(np.max(np.abs(phase)))*wavelength/(4*math.pi),places=6)

    def test_copernicus_product_name(self):
        self.assertEqual(tile_product("N38_E098"),"Copernicus_DSM_COG_10_N38_00_E098_00_DEM")


if __name__=="__main__":unittest.main()
