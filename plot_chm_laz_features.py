import re
from pathlib import Path
import numpy as np
import pandas as pd
import laspy
import rasterio
from scipy.stats import iqr, skew, kurtosis, entropy
from scipy.ndimage import generic_filter, uniform_filter, label
from math import ceil

"""
plot_chm_laz_features.py

Purpose:
    Extract plot-level LiDAR features from paired CHM and LAZ tiles.
    - CHM: canopy height statistics, canopy cover, gap metrics, roughness, simple texture
    - LAZ: height distribution, foliage height diversity, return/echo ratios, intensity stats

Expected naming:
    <prefix>_CHM_<res>m.tif
    <prefix>_LAZ_<res>m.laz
    e.g. VSN601001_CHM_20m.tif, VSN601001_LAZ_20m.laz

Output:
    - lichen_features.csv in the same folder (one row per plot/prefix_res)

Note:
    In the public repository, CLIPPED_DIR should point to a small demo folder
    containing a handful of clipped plot CHM/LAZ pairs.
"""

# ==== PATHS (edit when running locally) ====
# For the GitHub/demo version, keep this relative to the repository root
CLIPPED_DIR = Path("./demo_clipped_plots")
OUTPUT_CSV = CLIPPED_DIR / "lichen_features.csv"

# ==== PATHS & FILENAME PATTERNS (defaults) ====
# These defaults assume you are running from the repository root.
# CHM/LAZ clipped plots for the demo go in: data/demo_clipped_plots
# The script writes a single CSV with plot-level CHM + LAZ features.
CLIPPED_DIR = Path("data/demo_clipped_plots")
OUTPUT_CSV  = Path("outputs/plot_level_features/lichen_features.csv")

CHM_RE = re.compile(r"^(?P<prefix>.+?)_CHM_(?P<res>\d+m)\.(?i:tif)$")
LAZ_RE = re.compile(r"^(?P<prefix>.+?)_LAZ_(?P<res>\d+m)\.(?i:laz)$")

# ==== SMALL UTILITIES ====
def odd_approx(n: float) -> int:
    """Return the closest odd integer >= n."""
    base = int(n)
    return base if base % 2 == 1 else base + 1


def window_std(arr: np.ndarray, size_px: int) -> np.ndarray:
    """Local standard deviation with a square window of size_px pixels."""
    mu = uniform_filter(arr, size=size_px, mode="nearest")
    mu2 = uniform_filter(arr * arr, size=size_px, mode="nearest")
    var = np.maximum(mu2 - mu * mu, 0.0)
    return np.sqrt(var)


def fhd_from_z(z: np.ndarray, bin_size: float = 0.5) -> float:
    """Foliage Height Diversity (Shannon entropy of height distribution)."""
    if z.size == 0 or np.nanmax(z) <= 0:
        return 0.0
    bins = np.arange(0, np.nanmax(z) + bin_size, bin_size)
    if bins.size < 2:
        return 0.0
    hist, _ = np.histogram(z, bins=bins, density=True)
    return float(entropy(hist, base=2)) if np.any(hist) else 0.0


# ==== LAZ FEATURES ====
def extract_laz_features(laz_path: Path) -> dict:
    """Extract plot-level LAZ point cloud features."""
    try:
        las = laspy.read(str(laz_path))
        if len(las.points) == 0:
            return {}

        z = np.asarray(las.z, dtype=float)
        n = z.size
        if n == 0:
            return {}

        intensity = np.asarray(getattr(las, "intensity", np.array([], dtype=float)))
        return_num = np.asarray(getattr(las, "return_number", np.array([], dtype=int)))
        num_returns = np.asarray(
            getattr(
                las,
                "num_returns",
                getattr(las, "number_of_returns", np.array([], dtype=int)),
            )
        )

        feats = {
            "z_mean": float(np.nanmean(z)),
            "z_std": float(np.nanstd(z)),
            "z_max": float(np.nanmax(z)),
            "z_min": float(np.nanmin(z)),
            "z_iqr": float(iqr(z)),
            "z_p10": float(np.nanpercentile(z, 10)),
            "z_p25": float(np.nanpercentile(z, 25)),
            "z_p50": float(np.nanpercentile(z, 50)),
            "z_p75": float(np.nanpercentile(z, 75)),
            "z_p95": float(np.nanpercentile(z, 95)),
            "fhd_0p5m": fhd_from_z(z, 0.5),
            "ratio_below_0p5m": float(np.sum(z < 0.5) / n),
            "ratio_below_1m": float(np.sum(z < 1.0) / n),
            "mid_story_ratio": float(np.sum((z > 0.15) & (z <= 5.0)) / n),
            "high_canopy_ratio": float(np.sum(z > 5.0) / n),
            "canopy_cover_gt3m": float(np.sum(z > 3.0) / n),
            "canopy_cover_gt5m": float(np.sum(z > 5.0) / n),
            "num_points": int(n),
        }

        try:
            feats["z_skew"] = float(skew(z))
        except Exception:
            feats["z_skew"] = np.nan
        try:
            feats["z_kurtosis"] = float(kurtosis(z))
        except Exception:
            feats["z_kurtosis"] = np.nan

        # Return/echo ratios
        if return_num.size == n and num_returns.size == n:
            first = return_num == 1
            last = return_num == num_returns
            multi = num_returns > 1
            feats["ratio_first_returns"] = float(np.sum(first) / n)
            feats["ratio_last_returns"] = float(np.sum(last) / n)
            feats["ratio_multi_returns"] = float(np.sum(multi) / n)
            feats["first_returns_below_2m"] = float(np.sum((z < 2.0) & first) / n)
        else:
            feats["ratio_first_returns"] = np.nan
            feats["ratio_last_returns"] = np.nan
            feats["ratio_multi_returns"] = np.nan
            feats["first_returns_below_2m"] = np.nan

        # Intensity features
        if intensity.size == n and n > 0:
            feats["intensity_mean"] = float(np.nanmean(intensity))
            feats["intensity_std"] = float(np.nanstd(intensity))
            feats["intensity_cv"] = float(
                feats["intensity_std"] / (feats["intensity_mean"] + 1e-9)
            )
            feats["intensity_p10"] = float(np.nanpercentile(intensity, 10))
            feats["intensity_p50"] = float(np.nanpercentile(intensity, 50))
            feats["intensity_p90"] = float(np.nanpercentile(intensity, 90))

            low = intensity[z < 1.0]
            feats["intensity_low1m_mean"] = (
                float(np.nanmean(low)) if low.size else np.nan
            )
            feats["intensity_low1m_p90"] = (
                float(np.nanpercentile(low, 90)) if low.size else np.nan
            )
        else:
            feats.update(
                {
                    "intensity_mean": np.nan,
                    "intensity_std": np.nan,
                    "intensity_cv": np.nan,
                    "intensity_p10": np.nan,
                    "intensity_p50": np.nan,
                    "intensity_p90": np.nan,
                    "intensity_low1m_mean": np.nan,
                    "intensity_low1m_p90": np.nan,
                }
            )

        return feats

    except Exception:
        # For public code, we keep it silent and simply return empty on failure
        return {}


# ==== CHM FEATURES ====
def extract_chm_features(
    chm_path: Path, assumed_nodata_vals: tuple = (-9999, -3.4e38)
) -> dict:
    """Extract plot-level features from a CHM tile."""
    try:
        with rasterio.open(chm_path) as src:
            arr = src.read(1).astype(float)
            transform = src.transform
            pix_x = abs(transform.a)
            pix_y = abs(transform.e)
            pixel_size = float((pix_x + pix_y) / 2.0)

            mask = np.zeros_like(arr, dtype=bool)

            # Identify NoData
            if src.nodata is not None and not np.isnan(src.nodata):
                mask |= arr == src.nodata
            mask |= np.isnan(arr)
            if not np.any(mask):
                # Fallback to user-specified NoData candidates
                for v in assumed_nodata_vals:
                    mask |= arr == v

            valid = ~mask
            if not np.any(valid):
                return {}

            chm = np.where(valid, arr, np.nan)
            chm_data = chm[valid]

            feats = {
                "chm_mean": float(np.nanmean(chm_data)),
                "chm_std": float(np.nanstd(chm_data)),
                "chm_max": float(np.nanmax(chm_data)),
                "chm_min": float(np.nanmin(chm_data)),
                "chm_iqr": float(
                    np.nanpercentile(chm_data, 75)
                    - np.nanpercentile(chm_data, 25)
                ),
                "chm_num_pixels": int(np.sum(valid)),
            }
            try:
                feats["chm_skew"] = float(skew(chm_data))
            except Exception:
                feats["chm_skew"] = np.nan
            try:
                feats["chm_kurtosis"] = float(kurtosis(chm_data))
            except Exception:
                feats["chm_kurtosis"] = np.nan

            # Canopy cover / gap metrics for thresholds 3 m and 5 m
            for thr in (3.0, 5.0):
                canopy = (chm >= thr) & valid
                gaps = valid & ~canopy
                prop_canopy = np.sum(canopy) / np.sum(valid)
                prop_gaps = np.sum(gaps) / np.sum(valid)
                lab, nlab = label(gaps)
                max_gap = 0.0
                if nlab > 0:
                    counts = np.bincount(lab.ravel())[1:]
                    max_gap = int(np.max(counts)) * (pixel_size * pixel_size)

                feats[f"canopy_cover_gt{int(thr)}m"] = float(prop_canopy)
                feats[f"gap_fraction_lt{int(thr)}m"] = float(prop_gaps)
                feats[f"gap_count_lt{int(thr)}m"] = int(nlab)
                feats[f"gap_max_area_lt{int(thr)}m"] = float(max_gap)

            # Multi-scale roughness (local std in 1 m and 3 m windows)
            win1 = max(3, odd_approx(ceil(1.0 / max(pixel_size, 1e-6))))
            win3 = max(3, odd_approx(ceil(3.0 / max(pixel_size, 1e-6))))

            filled = np.where(valid, chm, np.nanmedian(chm_data))
            local_std_1m = window_std(filled, win1)[valid]
            local_std_3m = window_std(filled, win3)[valid]
            feats["chm_local_std_1m"] = float(np.nanmean(local_std_1m))
            feats["chm_local_std_3m"] = float(np.nanmean(local_std_3m))

            # Simple texture variance (3x3)
            local_var = generic_filter(
                np.nan_to_num(filled, nan=0.0), np.var, size=3, mode="nearest"
            )
            feats["chm_texture_variance"] = float(
                np.nanmean(local_var[valid])
            )

            return feats

    except Exception:
        return {}


# ==== SCAN & PAIR (by key = prefix_res) ====
def parse_key(name: str, which: str):
    """
    Parse CHM/LAZ filename and return (key, prefix, res) or (None, None, None).

    which in {'CHM','LAZ'}.
    key = "<prefix>_<res>", where <res> is like "20m".
    """
    m = CHM_RE.match(name) if which == "CHM" else LAZ_RE.match(name)
    if not m:
        return (None, None, None)
    prefix = m.group("prefix")
    res = m.group("res")
    return (f"{prefix}_{res}", prefix, res)


def main():
    if not CLIPPED_DIR.exists():
        print(f"[error] CLIPPED_DIR does not exist: {CLIPPED_DIR}")
        return

    chm_dict = {}
    for p in list(CLIPPED_DIR.glob("*_CHM_*m.tif")) + list(
        CLIPPED_DIR.glob("*_CHM_*m.TIF")
    ):
        k, prefix, res = parse_key(p.name, "CHM")
        if k:
            chm_dict[k] = p

    laz_dict = {}
    for p in list(CLIPPED_DIR.glob("*_LAZ_*m.laz")) + list(
        CLIPPED_DIR.glob("*_LAZ_*m.LAZ")
    ):
        k, prefix, res = parse_key(p.name, "LAZ")
        if k:
            laz_dict[k] = p

    print(f"[scan] CHM keys: {len(chm_dict)}   LAZ keys: {len(laz_dict)}")
    all_keys = sorted(set(chm_dict.keys()) | set(laz_dict.keys()))
    if not all_keys:
        print("[error] No CHM/LAZ files with the expected naming pattern were found.")
        return

    rows = []
    ok = only_chm = only_laz = both_invalid = 0
    total = len(all_keys)

    for i, key in enumerate(all_keys, 1):
        chm_path = chm_dict.get(key)
        laz_path = laz_dict.get(key)

        row = {
            "Name": key,  # prefix_res
            "CHM_file": chm_path.name if chm_path else "",
            "LAZ_file": laz_path.name if laz_path else "",
        }

        got_any = False

        # CHM features
        if chm_path:
            chm_feats = extract_chm_features(chm_path)
            if chm_feats:
                row.update(chm_feats)
                got_any = True
            else:
                only_laz += 1
        else:
            only_laz += 1

        # LAZ features
        if laz_path:
            laz_feats = extract_laz_features(laz_path)
            if laz_feats:
                row.update(laz_feats)
                got_any = True
            else:
                only_chm += 1
        else:
            only_chm += 1

        if got_any:
            rows.append(row)
            ok += 1
        else:
            both_invalid += 1

        if i % 20 == 0 or i == total:
            print(
                f"[progress] {i}/{total} done "
                f"(ok={ok}, only_CHM={only_chm}, only_LAZ={only_laz}, both_invalid={both_invalid})"
            )

    if ok > 0:
        OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(rows)
        first = ["Name", "CHM_file", "LAZ_file"]
        df = df[first + [c for c in df.columns if c not in first]]
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"[save] {ok} rows -> {OUTPUT_CSV}")
        print(
            f"[summary] only_CHM: {only_chm}, only_LAZ: {only_laz}, both_invalid: {both_invalid}"
        )
    else:
        print(
            f"[summary] only_CHM: {only_chm}, only_LAZ: {only_laz}, both_invalid: {both_invalid}"
        )
        print("[error] No valid CHM/LAZ feature data to save.")


if __name__ == "__main__":
    main()
