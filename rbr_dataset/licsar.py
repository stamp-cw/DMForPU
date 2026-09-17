"""Reserved real-InSAR interfaces for a later LiCSAR-assisted release."""


def load_licsar_data(*_args, **_kwargs):
    raise NotImplementedError("RBR v1 is DEM-driven; no LiCSAR product is presented as ground truth")


def extract_real_phase_statistics(*_args, **_kwargs):
    raise NotImplementedError("Enable after a versioned LiCSAR acquisition protocol is approved")


def extract_coherence_statistics(*_args, **_kwargs):
    raise NotImplementedError("Enable after a versioned LiCSAR acquisition protocol is approved")

