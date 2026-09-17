"""Create GIS boundary files for the RBR source regions and spatial splits."""
from __future__ import annotations

import json
from pathlib import Path

import shapefile
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "rbr" / "regions.yaml"
OUTPUT = ROOT / "data" / "rbr_sources" / "regions"
WGS84_ESRI_WKT = (
    'GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",'
    'SPHEROID["WGS_1984",6378137.0,298.257223563]],'
    'PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]]'
)
CN_NAMES = {
    "qilian_mountains": "祁连山",
    "lanzhou_loess": "兰州—定西黄土高原",
    "north_china_plain": "华北平原",
    "huainan_mining": "淮南矿区",
    "sichuan_yunnan_holdout": "川滇植被山区",
}
SPLITS = (
    ("train", 0.00, 0.64),
    ("validation", 0.68, 0.82),
    ("test_in_domain", 0.84, 1.00),
)


def ring(bbox: list[float]) -> list[list[float]]:
    xmin, ymin, xmax, ymax = map(float, bbox)
    return [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax], [xmin, ymin]]


def feature(region_id: int, region: dict, split: str, bbox: list[float]) -> dict:
    return {
        "type": "Feature",
        "properties": {
            "region_id": region_id,
            "name": region["name"],
            "cn_name": CN_NAMES[region["name"]],
            "role": region["role"],
            "split": split,
            "scene_type": region["scene_type"],
            "dem_tile": region["tile"],
            "source": "Copernicus_GLO-30_2021_AWS",
        },
        "geometry": {"type": "Polygon", "coordinates": [ring(bbox)]},
    }


def write_shapefile(stem: Path, features: list[dict]) -> None:
    with shapefile.Writer(str(stem), shapeType=shapefile.POLYGON, encoding="utf-8") as writer:
        writer.autoBalance = 1
        writer.field("REGION_ID", "N", 4, 0)
        writer.field("NAME", "C", 64)
        writer.field("CN_NAME", "C", 80)
        writer.field("ROLE", "C", 24)
        writer.field("SPLIT", "C", 24)
        writer.field("SCENE_TYPE", "C", 32)
        writer.field("DEM_TILE", "C", 16)
        writer.field("SOURCE", "C", 48)
        for item in features:
            props = item["properties"]
            writer.poly(item["geometry"]["coordinates"])
            writer.record(
                props["region_id"], props["name"], props["cn_name"], props["role"],
                props["split"], props["scene_type"], props["dem_tile"], props["source"],
            )
    stem.with_suffix(".prj").write_text(WGS84_ESRI_WKT, encoding="ascii")
    stem.with_suffix(".cpg").write_text("UTF-8\n", encoding="ascii")


def write_collection(stem: Path, features: list[dict]) -> None:
    collection = {
        "type": "FeatureCollection",
        "name": stem.name,
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
        "features": features,
    }
    stem.with_suffix(".geojson").write_text(
        json.dumps(collection, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_shapefile(stem, features)


def validate(stem: Path, expected: int) -> dict:
    reader = shapefile.Reader(str(stem.with_suffix(".shp")), encoding="utf-8")
    assert len(reader) == expected
    assert len(reader.shapes()) == len(reader.records()) == expected
    assert all(shape.shapeType == shapefile.POLYGON for shape in reader.shapes())
    assert stem.with_suffix(".prj").exists() and stem.with_suffix(".cpg").exists()
    parsed = json.loads(stem.with_suffix(".geojson").read_text(encoding="utf-8"))
    assert len(parsed["features"]) == expected
    return {"features": expected, "bbox": list(reader.bbox)}


def main() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    regions = config["regions"]
    OUTPUT.mkdir(parents=True, exist_ok=True)

    region_features = [feature(i, region, "full_region", region["bbox"]) for i, region in enumerate(regions)]
    split_features: list[dict] = []
    for i, region in enumerate(regions):
        xmin, ymin, xmax, ymax = map(float, region["bbox"])
        if region["role"] == "test_cross_region":
            split_features.append(feature(i, region, "test_cross_region", [xmin, ymin, xmax, ymax]))
            continue
        width = xmax - xmin
        for split, start, stop in SPLITS:
            split_features.append(feature(i, region, split, [xmin + start * width, ymin, xmin + stop * width, ymax]))

    regions_stem = OUTPUT / "rbr_regions"
    splits_stem = OUTPUT / "rbr_spatial_splits"
    write_collection(regions_stem, region_features)
    write_collection(splits_stem, split_features)
    summary = {
        "crs": "EPSG:4326",
        "source_config": str(CONFIG.relative_to(ROOT)),
        "regions": validate(regions_stem, 5),
        "spatial_splits": validate(splits_stem, 13),
        "split_rule": {
            "train": "longitude fraction [0.00, 0.64]",
            "validation": "longitude fraction [0.68, 0.82]",
            "test_in_domain": "longitude fraction [0.84, 1.00]",
            "test_cross_region": "entire Sichuan-Yunnan holdout region",
        },
    }
    (OUTPUT / "rbr_region_vectors.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
