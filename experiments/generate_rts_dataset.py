"""Generate the real-terrain RTS v1 phase-unwrapping dataset."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import time
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import requests
import yaml
from PIL import Image
from rasterio.windows import from_bounds
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "rts"
TWO_PI = 2.0 * math.pi
FAMILY_CODE = 733


def load_yaml(name: str) -> dict:
    return yaml.safe_load((CONFIG_DIR / name).read_text(encoding="utf-8"))


def wrap_phase(value: np.ndarray) -> np.ndarray:
    return np.angle(np.exp(1j * value)).astype(np.float32)


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tile_product(tile: str) -> str:
    latitude, longitude = tile.split("_")
    return f"Copernicus_DSM_COG_10_{latitude[:1]}{int(latitude[1:]):02d}_00_{longitude[:1]}{int(longitude[1:]):03d}_00_DEM"


def download_sources(dataset_cfg: dict, regions_cfg: dict) -> dict[str, Path]:
    cache = ROOT / dataset_cfg["source_cache"]
    cache.mkdir(parents=True, exist_ok=True)
    paths = {}
    for region in regions_cfg["regions"]:
        product = tile_product(region["tile"])
        url = f"https://copernicus-dem-30m.s3.amazonaws.com/{product}/{product}.tif"
        path = cache / f"{region['name']}.tif"
        if not path.exists():
            temporary = path.with_suffix(".tif.part")
            if temporary.exists(): temporary.unlink()
            print(f"download {region['name']}: {url}", flush=True)
            with requests.get(url, stream=True, timeout=120) as response:
                response.raise_for_status()
                with temporary.open("wb") as f:
                    for chunk in response.iter_content(4 << 20):
                        if chunk: f.write(chunk)
            temporary.replace(path)
        paths[region["name"]] = path
    provenance = {
        "source": regions_cfg["dem_source"],
        "accessed": time.strftime("%Y-%m-%d"),
        "registry": "https://registry.opendata.aws/copernicus-dem/",
        "license_notice": regions_cfg["license_notice"],
        "files": {name: {"file": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": sha256(path)} for name, path in paths.items()},
    }
    atomic_json(cache / "provenance.json", provenance)
    (cache / "README.md").write_text(
        "# RTS DEM sources\n\n" + regions_cfg["license_notice"] +
        "\n\nSource: https://registry.opendata.aws/copernicus-dem/\n", encoding="utf-8")
    return paths


def load_regions(dataset_cfg: dict, regions_cfg: dict, paths: dict[str, Path]) -> list[dict]:
    loaded = []
    for region_id, config in enumerate(regions_cfg["regions"]):
        with rasterio.open(paths[config["name"]]) as source:
            window = from_bounds(*config["bbox"], transform=source.transform).round_offsets().round_lengths()
            dem = source.read(1, window=window).astype(np.float32)
            transform = source.window_transform(window)
            nodata = source.nodata
        invalid = ~np.isfinite(dem)
        if nodata is not None: invalid |= dem == nodata
        if invalid.any():
            valid = dem[~invalid]
            if not len(valid): raise RuntimeError(f"DEM contains no valid cells: {config['name']}")
            dem[invalid] = float(np.median(valid))
        if min(dem.shape) < dataset_cfg["source_window_pixels"]:
            raise RuntimeError(f"DEM crop too small: {config['name']} {dem.shape}")
        loaded.append({"id": region_id, "config": config, "dem": dem, "transform": transform})
        print(f"loaded {config['name']}: {dem.shape}, {dem.min():.1f}..{dem.max():.1f} m", flush=True)
    return loaded


def resize(array: np.ndarray, size: int) -> np.ndarray:
    return np.asarray(Image.fromarray(array.astype(np.float32), mode="F").resize((size, size), Image.Resampling.BILINEAR), dtype=np.float32)


def seeded_rng(seed: int, split: str, index: int) -> tuple[np.random.Generator, int]:
    split_code = 101 if split == "train" else 211
    sequence = np.random.SeedSequence([seed, FAMILY_CODE, split_code, index])
    sample_seed = int(sequence.generate_state(1, dtype=np.uint64)[0])
    return np.random.default_rng(sequence), sample_seed


def choose_region(regions: list[dict], split: str, index: int, rng: np.random.Generator, cross_fraction: float) -> tuple[dict, str]:
    in_domain = [region for region in regions if region["config"]["role"] == "train_in_domain"]
    held_out = [region for region in regions if region["config"]["role"] == "test_cross_region"]
    if split == "test" and index % 10 < round(10 * cross_fraction):
        return held_out[index % len(held_out)], "test_cross_region"
    return in_domain[int(rng.integers(0, len(in_domain)))], "test_in_domain" if split == "test" else "train"


def source_patch(region: dict, partition: str, window_size: int, rng: np.random.Generator) -> tuple[np.ndarray, int, int, float, float]:
    dem = region["dem"]; height, width = dem.shape
    if partition == "train": low, high = 0.00, 0.64
    elif partition == "validation": low, high = 0.68, 0.82
    elif partition == "test_in_domain": low, high = 0.84, 1.00
    else: low, high = 0.00, 1.00
    col_min = int(low * width)
    col_max = min(width - window_size, int(high * width) - window_size)
    if col_max < col_min: raise RuntimeError(f"spatial split is narrower than source window for {region['config']['name']}")
    row = int(rng.integers(0, height - window_size + 1))
    col = int(rng.integers(col_min, col_max + 1))
    patch = dem[row:row+window_size, col:col+window_size]
    x, y = region["transform"] * (col + window_size / 2, row + window_size / 2)
    return patch, row, col, float(x), float(y)


def choose_weighted(mapping: dict, rng: np.random.Generator) -> str:
    names = list(mapping); probabilities = np.asarray(list(mapping.values()), np.float64)
    probabilities /= probabilities.sum()
    return names[int(rng.choice(len(names), p=probabilities))]


def deformation_phase(kind: str, size: int, wavelength: float, rng: np.random.Generator, span_cycles: list[float], dominant: bool) -> tuple[np.ndarray, float]:
    yy, xx = np.mgrid[-1:1:complex(size), -1:1:complex(size)]
    if kind == "terrain_only": return np.zeros((size, size), np.float32), 0.0
    target = TWO_PI * rng.uniform(span_cycles[0], span_cycles[1]) * (1.25 if dominant else 1.0)
    sign = -1.0 if rng.random() < .5 else 1.0
    cx, cy = rng.uniform(-.55, .55, 2); sx, sy = rng.uniform(.10, .42, 2)
    angle = rng.uniform(0, math.pi); ca, sa = math.cos(angle), math.sin(angle)
    xr, yr = ca*(xx-cx)+sa*(yy-cy), -sa*(xx-cx)+ca*(yy-cy)
    if kind == "gaussian_bowl": shape = np.exp(-((xx-cx)**2+(yy-cy)**2)/(2*sx**2))
    elif kind == "elliptical_bowl": shape = np.exp(-(xr*xr/(2*sx*sx)+yr*yr/(2*sy*sy)))
    elif kind == "multiple_centers":
        shape = np.zeros_like(xx)
        for _ in range(int(rng.integers(2, 5))):
            qx, qy = rng.uniform(-.7, .7, 2); qs = rng.uniform(.08, .30)
            shape += rng.uniform(-1, 1)*np.exp(-((xx-qx)**2+(yy-qy)**2)/(2*qs**2))
    elif kind == "linear_gradient": shape = ca*xx + sa*yy
    elif kind == "nonlinear_smooth": shape = np.sin(rng.uniform(1, 3)*math.pi*xx+angle)*np.cos(rng.uniform(1, 3)*math.pi*yy-angle)
    elif kind == "localized_high_gradient": shape = np.tanh(xr/rng.uniform(.025, .08))*np.exp(-(yr*yr)/(2*sy*sy))
    else: raise KeyError(kind)
    shape = shape - shape.mean(); extent = float(np.ptp(shape))
    phase = sign * target * shape / max(extent, 1e-8)
    peak_displacement = float(np.max(np.abs(phase))*wavelength/(4*math.pi))
    return phase.astype(np.float32), peak_displacement


def correlated_field(size: int, sigma: float, rng: np.random.Generator) -> np.ndarray:
    value = gaussian_filter(rng.normal(size=(size, size)), sigma=sigma, mode="reflect")
    value -= value.mean(); value /= max(float(value.std()), 1e-8)
    return value.astype(np.float32)


def render_scene(region: dict, source: np.ndarray, size: int, snr: float | None, rng: np.random.Generator, sim: dict) -> tuple[dict, dict]:
    dem = resize(source, size)
    satellites = sim["satellites"]; probs = np.asarray([x["probability"] for x in satellites]); probs /= probs.sum()
    satellite_id = int(rng.choice(len(satellites), p=probs)); satellite = satellites[satellite_id]
    wavelength = float(satellite["wavelength_m"])
    incidence = float(rng.uniform(*sim["geometry"]["incidence_angle_deg"])); slant = float(rng.uniform(*sim["geometry"]["slant_range_m"]))
    bmin, bmax = sim["geometry"]["perpendicular_baseline_abs_m"]
    bperp = float(rng.uniform(bmin, bmax) * (-1 if rng.random() < .5 else 1))
    deformation_kind = choose_weighted(sim["deformation"]["mixture"], rng)
    dominant = deformation_kind != "terrain_only" and rng.random() < (0.20/0.70)
    if dominant: bperp *= .18
    relative_height = dem - float(np.median(dem))
    topo = -4*math.pi*bperp*relative_height/(wavelength*slant*math.sin(math.radians(incidence)))
    deform, peak_displacement = deformation_phase(deformation_kind, size, wavelength, rng, sim["deformation"]["phase_span_cycles"], dominant)
    phase = topo + deform; phase -= phase.mean()
    max_phase = float(sim["geometry"]["max_absolute_phase_rad"]); limit = max_phase - math.pi
    scale = min(1.0, limit/max(float(np.max(np.abs(phase))), 1e-8))
    topo *= scale; deform *= scale; phase *= scale; bperp *= scale; peak_displacement *= scale
    margin = max_phase - float(np.max(np.abs(phase))); offset = float(rng.uniform(-min(math.pi, margin), min(math.pi, margin)))
    phase = (phase + offset).astype(np.float32)
    difficulty = choose_weighted(sim["coherence"]["difficulty_mixture"], rng)
    band = sim["coherence"]["bands"][difficulty]; base = float(rng.uniform(*band))
    pixel_spacing = 30.0 * source.shape[0] / size
    gy, gx = np.gradient(dem, pixel_spacing); slope = np.hypot(gx, gy)
    slope_norm = np.clip(slope/max(float(np.quantile(slope, .95)), 1e-8), 0, 1)
    sigma_field = float(rng.uniform(*sim["coherence"]["correlated_field_sigma_pixels"])) * size/128
    spatial = correlated_field(size, max(.5, sigma_field), rng)
    type_bias = {"flat_plain": .07, "mining": .02, "steep_mountain": -.03, "loess_gully": -.02, "vegetated_mountain": -.10}.get(region["config"]["scene_type"], 0.)
    coherence = np.clip(base + .07*spatial - .16*slope_norm + type_bias, sim["noise"]["coherence_floor"], .98).astype(np.float32)
    clean_wrapped = wrap_phase(phase)
    if snr is None:
        noisy_wrapped = clean_wrapped.copy()
    else:
        nr = correlated_field(size, max(.45, rng.uniform(.45, 1.8)*size/128), rng)
        ni = correlated_field(size, max(.45, rng.uniform(.45, 1.8)*size/128), rng)
        complex_noise = (nr + 1j*ni)/math.sqrt(2)
        nominal_sigma = math.sqrt(float(sim["noise"]["signal_power"])/(10**(float(snr)/10)))
        local_scale = np.sqrt(np.maximum(1/coherence**2-1, 0)+.05)
        noisy_complex = np.exp(1j*phase) + nominal_sigma*local_scale*complex_noise
        noisy_wrapped = np.angle(noisy_complex).astype(np.float32)
    noise_map = wrap_phase(noisy_wrapped-clean_wrapped)
    mode = "terrain_only" if deformation_kind == "terrain_only" else "deformation_dominant" if dominant else "terrain_plus_deformation"
    arrays = {"psi": noisy_wrapped, "phi": phase, "phi_wrapped_clean": clean_wrapped, "coherence": coherence.astype(np.float16), "dem": dem, "deformation_phase": deform.astype(np.float32), "noise_map": noise_map.astype(np.float16)}
    metadata = {"satellite_id": satellite_id, "wavelength": wavelength, "incidence_angle_deg": incidence, "slant_range_m": slant, "b_perp_m": bperp, "phase_offset": offset, "difficulty": difficulty, "deformation_type": deformation_kind, "scene_mode": mode, "peak_displacement_m": peak_displacement}
    return arrays, metadata


def metadata_maps(sim: dict) -> dict:
    return {"difficulty": {name: i for i, name in enumerate(sim["coherence"]["difficulty_mixture"])}, "deformation_type": {name: i for i, name in enumerate(sim["deformation"]["mixture"])}, "scene_mode": {name: i for i, name in enumerate(("terrain_only", "terrain_plus_deformation", "deformation_dominant"))}, "partition": {name: i for i, name in enumerate(("train", "validation", "test_in_domain", "test_cross_region"))}}


def snr_vector(count: int, fixed_snr: int | None, seed: int) -> np.ndarray:
    if fixed_snr is not None: return np.full(count, fixed_snr, np.float32)
    rng = np.random.default_rng(np.random.SeedSequence([seed, FAMILY_CODE, 991]))
    levels = np.asarray(load_yaml("dataset.yaml")["train_snrs_db"], np.float32)
    return rng.choice(levels, count, replace=True).astype(np.float32)


def write_file(path: Path, size: int, split: str, count: int, fixed_snr: int | None, seed: int, regions: list[dict], dataset_cfg: dict, sim: dict, maps: dict, noise_free: bool = False) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True); temporary = path.with_suffix(".h5.tmp")
    if temporary.exists(): temporary.unlink()
    snrs = np.full(count, np.inf, np.float32) if noise_free else snr_vector(count, fixed_snr, seed); chunk = (min(16, count), size, size)
    stats = {"phase_span": [], "coherence_mean": [], "dem_range": [], "gradient_mean": [], "difficulty": {}, "region": {}, "deformation": {}}
    validation_count = min(dataset_cfg["validation_count_within_train"], max(1, round(.1*count)))
    with h5py.File(temporary, "w") as f:
        datasets = {
            "psi": f.create_dataset("psi", (count,size,size), "f4", chunks=chunk, compression=dataset_cfg["compression"]),
            "phi": f.create_dataset("phi", (count,size,size), "f4", chunks=chunk, compression=dataset_cfg["compression"]),
            "phi_wrapped_clean": f.create_dataset("phi_wrapped_clean", (count,size,size), "f4", chunks=chunk, compression=dataset_cfg["compression"]),
            "coherence": f.create_dataset("coherence", (count,size,size), "f2", chunks=chunk, compression=dataset_cfg["compression"]),
            "dem": f.create_dataset("dem", (count,size,size), "f4", chunks=chunk, compression=dataset_cfg["compression"]),
            "deformation_phase": f.create_dataset("deformation_phase", (count,size,size), "f4", chunks=chunk, compression=dataset_cfg["compression"]),
            "noise_map": f.create_dataset("noise_map", (count,size,size), "f2", chunks=chunk, compression=dataset_cfg["compression"]),
        }
        scalar_specs = {"snr":"f4", "scene_id":"i8", "random_seed":"u8", "region_id":"u1", "partition":"u1", "difficulty":"u1", "deformation_type":"u1", "scene_mode":"u1", "satellite_id":"u1", "source_row":"i4", "source_col":"i4", "center_lon":"f8", "center_lat":"f8", "wavelength_m":"f4", "incidence_angle_deg":"f4", "slant_range_m":"f4", "b_perp_m":"f4", "phase_offset":"f4", "peak_displacement_m":"f4"}
        scalars = {name: f.create_dataset(name, (count,), dtype=dtype) for name,dtype in scalar_specs.items()}
        f.attrs.update(family="RTS", source_name="Copernicus GLO-30 real-terrain simulation", split=split, image_size=size, seed=seed, generator_version="1.1", noise_model="none" if noise_free else sim["noise"]["mode"], noise_free=noise_free, snr_label="clean" if noise_free else str(fixed_snr), metadata_maps=json.dumps(maps), region_maps=json.dumps({r["config"]["name"]:r["id"] for r in regions}), satellite_maps=json.dumps({s["name"]:i for i,s in enumerate(sim["satellites"])}))
        for index in range(count):
            rng, sample_seed = seeded_rng(seed, split, index)
            region, domain = choose_region(regions, split, index, rng, dataset_cfg["test_cross_region_fraction"])
            partition = ("validation" if split == "train" and index >= count-validation_count else "train") if split == "train" else domain
            source, row, col, lon, lat = source_patch(region, partition, dataset_cfg["source_window_pixels"], rng)
            arrays, meta = render_scene(region, source, size, None if noise_free else float(snrs[index]), rng, sim)
            for name, value in arrays.items(): datasets[name][index] = value
            values = {"snr":snrs[index], "scene_id":index, "random_seed":sample_seed, "region_id":region["id"], "partition":maps["partition"][partition], "difficulty":maps["difficulty"][meta["difficulty"]], "deformation_type":maps["deformation_type"][meta["deformation_type"]], "scene_mode":maps["scene_mode"][meta["scene_mode"]], "satellite_id":meta["satellite_id"], "source_row":row, "source_col":col, "center_lon":lon, "center_lat":lat, "wavelength_m":meta["wavelength"], "incidence_angle_deg":meta["incidence_angle_deg"], "slant_range_m":meta["slant_range_m"], "b_perp_m":meta["b_perp_m"], "phase_offset":meta["phase_offset"], "peak_displacement_m":meta["peak_displacement_m"]}
            for name, value in values.items(): scalars[name][index] = value
            stats["phase_span"].append(float(np.ptp(arrays["phi"]))); stats["coherence_mean"].append(float(np.mean(arrays["coherence"]))); stats["dem_range"].append(float(np.ptp(arrays["dem"]))); stats["gradient_mean"].append(float(np.mean(np.hypot(np.gradient(arrays["phi"],axis=0),np.gradient(arrays["phi"],axis=1)))))
            for key, value in (("difficulty",meta["difficulty"]),("region",region["config"]["name"]),("deformation",meta["deformation_type"])): stats[key][value]=stats[key].get(value,0)+1
            if (index+1)%250==0 or index+1==count: print(f"RTS{size} {path.name}: {index+1}/{count}", flush=True)
    temporary.replace(path)
    audit = {"samples":count,"shape":[count,size,size],"file":str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),"bytes":path.stat().st_size,"sha256":sha256(path),"phase_span_mean":float(np.mean(stats["phase_span"])),"coherence_mean":float(np.mean(stats["coherence_mean"])),"dem_range_mean_m":float(np.mean(stats["dem_range"])),"phase_gradient_mean":float(np.mean(stats["gradient_mean"])),"difficulty_counts":stats["difficulty"],"region_counts":stats["region"],"deformation_counts":stats["deformation"]}
    return audit


def validate_dataset(output: Path, sizes: list[int], train_count: int, test_count: int, seed: int) -> dict:
    rng = np.random.default_rng(seed+8001); checks=[]; hashes={}; duplicates=0; failures=[]
    files = [output/f"RTS{size}"/name for size in sizes for name in ["train.h5","test_clean.h5",*[f"test_{s}dB.h5" for s in load_yaml("dataset.yaml")["test_snrs_db"]]]]
    candidates = [(path, index) for path in files for index in range(train_count if path.name=="train.h5" else test_count)]
    chosen = rng.choice(len(candidates), min(120, len(candidates)), replace=False)
    for choice in chosen:
        path, index = candidates[int(choice)]
        with h5py.File(path,"r") as f:
            psi=f["psi"][index];phi=f["phi"][index];clean=f["phi_wrapped_clean"][index];gamma=f["coherence"][index];dem=f["dem"][index]
        consistency=float(np.max(np.abs(wrap_phase(phi)-clean))); finite=bool(all(np.isfinite(x).all() for x in (psi,phi,clean,gamma,dem))); wrapped=bool(psi.min()>=-math.pi-1e-6 and psi.max()<math.pi+1e-6); coherence=bool(gamma.min()>0 and gamma.max()<=1)
        digest=hashlib.sha1(phi.tobytes()).hexdigest(); identity=(path.parent.name,"train" if path.name=="train.h5" else "test",index)
        duplicates += int(digest in hashes and hashes[digest] != identity); hashes[digest]=identity
        if not(finite and wrapped and coherence and consistency<2e-5):failures.append({"file":str(path),"index":index,"finite":finite,"wrapped":wrapped,"coherence":coherence,"consistency":consistency})
        checks.append(consistency)
    # Same test ground truth and metadata must be invariant across SNR files.
    cross_snr=[]; clean_exact=[]
    for size in sizes:
        with h5py.File(output/f"RTS{size}"/"test_0dB.h5","r") as a,h5py.File(output/f"RTS{size}"/"test_30dB.h5","r") as b,h5py.File(output/f"RTS{size}"/"test_clean.h5","r") as clean:
            ids=sorted(rng.choice(test_count,min(20,test_count),replace=False));cross_snr.append(float(max(np.max(np.abs(a["phi"][ids]-b["phi"][ids])),np.max(np.abs(a["phi"][ids]-clean["phi"][ids])))))
            clean_exact.append(bool(np.isposinf(clean["snr"][:]).all() and np.count_nonzero(clean["noise_map"][:])==0 and np.array_equal(clean["psi"][:],clean["phi_wrapped_clean"][:])))
    passed=not failures and duplicates==0 and max(cross_snr)<1e-7 and all(clean_exact)
    report={"state":"pass" if passed else "fail","random_samples_checked":len(chosen),"failures":failures,"unexpected_duplicate_clean_scenes":duplicates,"expected_cross_snr_scene_matches_ignored":True,"max_wrap_consistency_error":max(checks),"max_cross_snr_gt_difference":max(cross_snr),"clean_exact":all(clean_exact)}
    atomic_json(output/"RTS_QA_REPORT.json",report)
    if report["state"]!="pass":raise RuntimeError(f"RTS QA failed: {report}")
    return report


def visualize(output: Path) -> Path:
    path=output/"RTS128"/"test_10dB.h5"
    with h5py.File(path,"r") as f: indices=list(range(min(5,len(f["psi"]))))
    with h5py.File(path,"r") as f: dem=f["dem"][indices];phi=f["phi"][indices];clean=f["phi_wrapped_clean"][indices];psi=f["psi"][indices];gamma=f["coherence"][indices]
    fig,axes=plt.subplots(len(indices),5,figsize=(13,2.6*len(indices)),squeeze=False,constrained_layout=True);headers=("DEM (m)","Unwrapped GT","Clean wrapped","Noisy wrapped 10 dB","Coherence")
    for col,title in enumerate(headers):axes[0,col].set_title(title)
    for row in range(len(indices)):
        for col,(value,cmap) in enumerate(zip((dem[row],phi[row],clean[row],psi[row],gamma[row]),("terrain","turbo","twilight_shifted","twilight_shifted","viridis"))):axes[row,col].imshow(value,cmap=cmap);axes[row,col].set_xticks([]);axes[row,col].set_yticks([])
    preview=output/"RTS_preview.png";fig.savefig(preview,dpi=170);plt.close(fig);return preview


def update_global_manifest(output: Path, entries: dict, dataset_cfg: dict, regions_cfg: dict, qa: dict, elapsed: float) -> None:
    path=output/"DATASETS_V2.json"; manifest=json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"schema_version":1,"families":{}}
    manifest["families"]["RTS"]={"status":"generated","generated":True,"version":"1.1","source":"Copernicus GLO-30 real terrain","sizes":entries,"protocol":{"train_count":dataset_cfg["train_count"],"validation_count_within_train":dataset_cfg["validation_count_within_train"],"test_count_per_condition":dataset_cfg["test_count_per_snr"],"train_snrs":dataset_cfg["train_snrs_db"],"test_snrs":dataset_cfg["test_snrs_db"],"test_conditions":["clean",*dataset_cfg["test_snrs_db"]],"spatial_split":True,"test_cross_region_fraction":dataset_cfg["test_cross_region_fraction"],"noise":"complex-domain, spatial coherence modulated"},"qa":qa,"elapsed_seconds":elapsed,"license_notice":regions_cfg["license_notice"]}
    atomic_json(path,manifest)


def generate(output: Path, train_count: int, test_count: int, sizes: list[int], seed: int, update_manifest: bool) -> None:
    started=time.time();dataset_cfg=load_yaml("dataset.yaml");sim=load_yaml("simulation.yaml");regions_cfg=load_yaml("regions.yaml");dataset_cfg.update(train_count=train_count,test_count_per_snr=test_count,seed=seed)
    paths=download_sources(dataset_cfg,regions_cfg);regions=load_regions(dataset_cfg,regions_cfg,paths);maps=metadata_maps(sim);entries={}
    for size in sizes:
        folder=output/f"RTS{size}";folder.mkdir(parents=True,exist_ok=True);files={}
        files["train"]=write_file(folder/"train.h5",size,"train",train_count,None,seed,regions,dataset_cfg,sim,maps)
        files["test_clean"]=write_file(folder/"test_clean.h5",size,"test",test_count,None,seed,regions,dataset_cfg,sim,maps,noise_free=True)
        for snr in dataset_cfg["test_snrs_db"]:files[f"test_{snr}dB"]=write_file(folder/f"test_{snr}dB.h5",size,"test",test_count,snr,seed,regions,dataset_cfg,sim,maps)
        entries[str(size)]=files;atomic_json(folder/"manifest.json",files)
    qa=validate_dataset(output,sizes,train_count,test_count,seed);preview=visualize(output) if 128 in sizes else None
    summary={"schema_version":1,"generator":"experiments/generate_rts_dataset.py","seed":seed,"config_files":["configs/rts/dataset.yaml","configs/rts/regions.yaml","configs/rts/simulation.yaml"],"test_conditions":["clean",*dataset_cfg["test_snrs_db"]],"sizes":entries,"qa":qa,"preview":None if preview is None else str(preview.relative_to(ROOT)) if preview.is_relative_to(ROOT) else str(preview),"elapsed_seconds":time.time()-started}
    atomic_json(output/"RTS_DATASET.json",summary)
    if update_manifest:update_global_manifest(output,entries,dataset_cfg,regions_cfg,qa,time.time()-started)
    print(f"RTS complete in {(time.time()-started)/60:.1f} min: {output}",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--output",type=Path,default=ROOT/"data");parser.add_argument("--train-count",type=int);parser.add_argument("--test-count",type=int);parser.add_argument("--sizes",nargs="+",type=int,choices=(32,64,128));parser.add_argument("--seed",type=int);parser.add_argument("--update-manifest",action="store_true");args=parser.parse_args();cfg=load_yaml("dataset.yaml")
    generate(args.output.resolve(),args.train_count or cfg["train_count"],args.test_count or cfg["test_count_per_snr"],args.sizes or cfg["sizes"],args.seed or cfg["seed"],args.update_manifest)
