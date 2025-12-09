# -*- coding: utf-8 -*-
"""
Batch ITC detection & delineation when CHM and LAZ are in the SAME folder.
- Pairs files like VSN601001_CHM_20m.tif  <->  VSN601001_LAZ_20m.laz (LAZ optional)
- Detection: multi-scale LoG + LHD + Hessian filter + 2-stage NMS
- Delineation: marker-controlled watershed (Sobel gradient)
- Exports (per plot): treetops.csv, crowns_labels.tif, crowns.gpkg (optional), tree_table.csv
- Exports (global): master_itc_features.csv (slim plot-level features)

Author: (your name), 2025-09-20
"""

import os, re, csv, time, warnings, glob, math, argparse
import numpy as np
import rasterio
from scipy.ndimage import gaussian_laplace, gaussian_filter, grey_closing, uniform_filter
from scipy.spatial import cKDTree
from skimage.feature import peak_local_max
from skimage.morphology import disk, closing
from skimage import exposure
from skimage.segmentation import watershed
from skimage.filters import sobel

warnings.filterwarnings("ignore", category=UserWarning)

# ===================== USER CONFIG (defaults) =====================
# These defaults assume you are running from the repository root.
# They can still be overridden by command-line arguments (if provided).
DATA_DIR = "data/demo_clipped_plots"      # folder with demo CHM(.tif) & LAZ(.laz) clipped plots
OUT_ROOT = "outputs/itc_delineation"      # root folder for ITC outputs
DO_POLY  = True                           # export crown polygons as GeoPackage (if geopandas available)
VERBOSE  = True
DEBUG_DUMPS = False                       # set True to save intermediate rasters

# ---- Detection parameters (tuned for ~0.5 m / 20×20 m plots) ----
FILL_PITS   = True
PIT_RADIUS_M= 1.2
PIT_DEPTH_M = 0.8
USE_CLAHE   = True

MASK_MIN_H  = 3.5
MIN_H       = 4.0

SIGMAS_M = (0.6, 0.9, 1.3, 1.8, 2.6, 3.2)
PCTL_SMALL, PCTL_LARGE = 93.0, 96.0
LHD_THR, LHD_R1, LHD_R2 = 0.9, 2.0, 5.0
NMS_K = 2.2
USE_PLATEAU = True
PLATEAU_PICK = "lhd"  # or 'resp'

# ---- Segmentation parameters ----
SLOPE_SMOOTH_SIGMA_PX = 1.0
WATERSHED_COMPACTNESS = 8.0
MIN_CROWN_AREA_M2     = 2.0
# r_max(h) = clip(0.8 + 0.25*h, 1.2, 5.0)

# ---- Slim features selection ----
SLIM_FEATURES = [
    "plot_id","n_trees","mean_nnd_m","cv_nnd",
    "cover_ratio","mean_radius_m","cv_radius","pct_oversize","pct_edge_trees","mean_peak_h"
]

# ===================== HELPERS =====================
def log(msg):
    if VERBOSE:
        print(msg)

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def quick_stats(name, arr, mask=None):
    a = arr if mask is None else arr[mask]
    a = a[np.isfinite(a)]
    if a.size == 0:
        return f"{name}: empty"
    q = np.percentile(a, [5, 50, 95])
    return f"{name}: n={a.size}, min={a.min():.3f}, p50={q[1]:.3f}, p95={q[2]:.3f}, max={a.max():.3f}"

def save_tif_like(src_ds, out_path, array, dtype="float32", nodata=None):
    ensure_dir(os.path.dirname(out_path))
    prof = src_ds.profile.copy()
    prof.update(dtype=dtype, count=1, compress="lzw")
    if nodata is not None:
        prof.update(nodata=nodata)
    with rasterio.open(out_path, "w", **prof) as dst:
        dst.write(array.astype(dtype), 1)
    log(f"[save] {out_path}")

def read_chm(chm_path):
    src = rasterio.open(chm_path)
    chm = src.read(1).astype("float32")
    chm[np.isnan(chm)] = 0.0
    chm[chm < 0] = 0.0
    px = float(src.res[0])
    log(f"[info] CHM read: {os.path.basename(chm_path)}, shape={chm.shape}, res={px} m")
    return src, chm, px

def fill_small_pits(chm, px, radius_m=1.5, max_depth_m=1.0):
    size_px = max(int(round(radius_m/px))*2+1, 3)
    closed = grey_closing(chm, size=(size_px, size_px))
    delta = closed - chm
    out = np.where(delta <= max_depth_m, closed, chm + max_depth_m).astype("float32")
    return out

def height_mask(chm, min_h=0.5, morph_close_px=3):
    mask = chm > float(min_h)
    mask = closing(mask, footprint=disk(morph_close_px))
    return mask

def lhd_map(chm, r1_m, r2_m, px):
    sigma2 = max(r2_m / px / 2.355, 0.2)
    bg = gaussian_filter(chm, sigma=sigma2)
    return chm - bg

def apply_clahe(chm, mask):
    vals = chm[mask]
    if vals.size == 0:
        return chm
    vmin, vmax = float(vals.min()), float(vals.max())
    if vmax - vmin < 1e-6:
        return chm
    v_norm = (vals - vmin) / (vmax - vmin)
    v_eq = exposure.equalize_adapthist(v_norm, clip_limit=0.01, nbins=256)
    out = chm.copy()
    out[mask] = v_eq * (vmax - vmin) + vmin
    return out

def canopy_mask_from_height(chm, px, hmin=5.0, min_area_m2=16.0, close_px=3):
    mask_h = chm >= float(hmin)
    mask_h = closing(mask_h, footprint=disk(close_px))
    from scipy.ndimage import label as cc_label
    lab, num = cc_label(mask_h)
    keep = np.zeros_like(mask_h, dtype=bool)
    if num > 0:
        area_thr_pix = max(1, int(round(min_area_m2 / (px * px))))
        sizes = np.bincount(lab.ravel())
        large_ids = np.where(sizes >= area_thr_pix)[0]
        large_ids = large_ids[large_ids != 0]
        if large_ids.size > 0:
            keep = np.isin(lab, large_ids)
    return keep

def roughness_mask(chm, px, win_m=2.0, std_thr=0.28):
    k = max(1, int(round(win_m / px)))
    mean = uniform_filter(chm, size=k, mode="nearest")
    mean2 = uniform_filter(chm*chm, size=k, mode="nearest")
    var = np.maximum(mean2 - mean*mean, 0.0)
    std = np.sqrt(var, dtype=np.float32)
    return std >= float(std_thr)

def per_scale_threshold(resp, mask, pctl_default, min_support=5000, relax=2.0):
    vals = resp[mask & (resp > 0)]
    if vals.size == 0:
        return np.inf
    pctl = pctl_default if vals.size >= min_support else max(50.0, pctl_default - relax)
    return float(np.percentile(vals, pctl))

def detect_multiscale(chm, mask, px):
    if USE_CLAHE:
        chm = apply_clahe(chm, mask)
        log(quick_stats("CHM_after_CLAHE", chm, mask))
    lhd = lhd_map(chm, LHD_R1, LHD_R2, px)
    log(quick_stats("LHD", lhd, mask))

    sig_px = [max(s/px, 0.5) for s in SIGMAS_M]
    R = []
    for s in sig_px:
        resp = -gaussian_laplace(chm, sigma=s)
        R.append((s**2) * resp.astype("float32"))
    R = np.stack(R, axis=0)

    small_large_boundary_m = 1.0
    candidates = []  # (col,row,idx,sigma_m,height,score)

    for i, s_m in enumerate(SIGMAS_M):
        resp = R[i]
        pctl = PCTL_SMALL if s_m <= small_large_boundary_m else PCTL_LARGE
        thr = per_scale_threshold(resp, mask, pctl_default=pctl)
        if not np.isfinite(thr) or thr <= 0:
            continue
        resp_thr = resp.copy()
        resp_thr[~mask] = 0.0
        resp_thr[resp_thr < thr] = 0.0
        md = max(int(1.6 * (s_m / px)), 1)

        if USE_PLATEAU:
            coords = peak_local_max(resp_thr, min_distance=md, threshold_abs=thr, exclude_border=False)
            peaks_map = np.zeros_like(resp_thr, dtype=bool)
            if coords.size > 0:
                peaks_map[tuple(coords.T)] = True
            from scipy.ndimage import label as cc_label
            lab, num = cc_label(peaks_map)
            coords2 = []
            plateau_area = {}
            if num > 0:
                score_img = lhd if PLATEAU_PICK.lower() == "lhd" else resp_thr
                for lab_id in range(1, num + 1):
                    rr, cc = np.where(lab == lab_id)
                    if rr.size == 0:
                        continue
                    idx0 = np.argmax(score_img[rr, cc])
                    r0, c0 = int(rr[idx0]), int(cc[idx0])
                    coords2.append([r0, c0])
                    plateau_area[(r0, c0)] = rr.size
            coords = np.asarray(coords2, dtype=int) if len(coords2) > 0 else np.empty((0, 2), dtype=int)
        else:
            coords = peak_local_max(resp_thr, min_distance=md, threshold_abs=thr, exclude_border=False)
            plateau_area = {}

        s_pix = max(s_m / px, 0.8)
        Ixx = gaussian_filter(chm, sigma=s_pix, order=(2, 0))
        Iyy = gaussian_filter(chm, sigma=s_pix, order=(0, 2))
        Ixy = gaussian_filter(chm, sigma=s_pix, order=(1, 1))

        for (row, col) in coords:
            hpeak = float(chm[row, col])
            if hpeak < float(MIN_H):
                continue
            local_lhd_thr = LHD_THR
            if plateau_area.get((row, col), 0) >= 30:
                local_lhd_thr = max(0.8, LHD_THR - 0.2)
            if LHD_THR is not None and lhd[row, col] < float(local_lhd_thr):
                continue
            best_i = int(np.argmax(R[:, row, col]))
            if best_i != i:
                continue
            trace = Ixx[row, col] + Iyy[row, col]
            det = Ixx[row, col] * Iyy[row, col] - Ixy[row, col] * Ixy[row, col]
            disc = max(trace * trace - 4.0 * det, 0.0)
            lam1 = 0.5 * (trace + np.sqrt(disc))
            lam2 = 0.5 * (trace - np.sqrt(disc))
            if not (lam1 < 0 and lam2 < 0):
                continue
            blobness = min(abs(lam1), abs(lam2)) / (max(abs(lam1), abs(lam2)) + 1e-6)
            if blobness < 0.50:
                continue
            score = float(resp[row, col])
            candidates.append((int(col), int(row), i, float(s_m), hpeak, score))

    if len(candidates) == 0:
        return [], R, lhd, chm

    # NMS #1
    pts = np.array(candidates, dtype=object)
    order = np.argsort([-p[-1] for p in pts])
    pts_sorted = pts[order]
    xs = np.array([p[0] for p in pts_sorted], dtype=np.float32)
    ys = np.array([p[1] for p in pts_sorted], dtype=np.float32)
    sigmas_m = np.array([p[3] for p in pts_sorted], dtype=np.float32)
    r_pix = (NMS_K * (sigmas_m / px)).astype(np.float32)
    tree = cKDTree(np.c_[xs, ys])
    suppressed = np.zeros(xs.shape[0], dtype=bool)
    kept_idx = []
    for i in range(xs.shape[0]):
        if suppressed[i]:
            continue
        kept_idx.append(i)
        nbrs = tree.query_ball_point([xs[i], ys[i]], r=float(r_pix[i]))
        suppressed[nbrs] = True
        suppressed[i] = False
    kept = [tuple(pts_sorted[i]) for i in kept_idx]

    # NMS #2
    if len(kept) > 0:
        pts2 = np.array(kept, dtype=object)
        xs2  = np.array([p[0] for p in pts2], dtype=np.float32)
        ys2  = np.array([p[1] for p in pts2], dtype=np.float32)
        sig2 = np.array([p[3] for p in pts2], dtype=np.float32)
        r2   = (0.8 * NMS_K * (sig2 / px)).astype(np.float32)
        tree2 = cKDTree(np.c_[xs2, ys2])
        taken = np.zeros(xs2.shape[0], dtype=bool)
        merged = []
        order2 = np.argsort([-p[-1] for p in pts2])
        for idx in order2:
            if taken[idx]:
                continue
            nbrs = tree2.query_ball_point([xs2[idx], ys2[idx]], r=float(r2[idx]))
            taken[nbrs] = True
            merged.append(tuple(pts2[idx]))
        kept = merged
    return kept, R, lhd, chm

def rmax_from_height(h):
    return float(np.clip(0.8 + 0.25*h, 1.2, 5.0))

def label_edges_touch(lab):
    edge = np.zeros_like(lab, dtype=bool)
    edge[0, :] = True
    edge[-1, :] = True
    edge[:, 0] = True
    edge[:, -1] = True
    touch_ids = np.unique(lab[edge])
    return set(int(x) for x in touch_ids if x != 0)

def crowns_via_watershed(chm, treetops, mask, px, src, out_lbl_tif=None):
    markers = np.zeros(chm.shape, dtype=np.int32)
    for i, (col, row, *_rest) in enumerate(treetops, start=1):
        markers[int(row), int(col)] = i
    grad = sobel(gaussian_filter(chm, sigma=SLOPE_SMOOTH_SIGMA_PX))
    labels = watershed(image=grad, markers=markers, mask=mask, compactness=WATERSHED_COMPACTNESS)
    if out_lbl_tif:
        save_tif_like(src, out_lbl_tif, labels, dtype="uint32", nodata=0)

    touch_ids = label_edges_touch(labels)
    recs = []
    for lab_id in range(1, labels.max()+1):
        pixmask = labels == lab_id
        if not np.any(pixmask):
            continue
        area_m2 = float(pixmask.sum() * px * px)
        if area_m2 < MIN_CROWN_AREA_M2:
            labels[pixmask] = 0
            continue
        peak_row, peak_col, _, sigma_m, hpeak, score = treetops[lab_id-1]
        mean_h = float(chm[pixmask].mean())
        max_h  = float(chm[pixmask].max())
        radius_eq = float(np.sqrt(area_m2/np.pi))
        r_max = rmax_from_height(hpeak)
        oversize = radius_eq > (1.15 * r_max)
        recs.append({
            "tree_id": lab_id, "row": int(peak_row), "col": int(peak_col),
            "peak_h": float(hpeak), "sigma_m": float(sigma_m), "score": float(score),
            "area_m2": area_m2, "radius_eq_m": radius_eq, "r_max_m": r_max,
            "oversize": int(oversize), "edge_flag": int(lab_id in touch_ids),
            "mean_h": mean_h, "max_h": max_h
        })
    return labels, recs

def pix_to_xy(transform, col, row):
    x = transform.c + col*transform.a + row*transform.b
    y = transform.f + col*transform.d + row*transform.e
    return float(x), float(y)

def export_tops_csv(out_csv, treetops, transform):
    fields = ["x","y","col","row","height","sigma_m","score","crown_r_est_m"]
    ensure_dir(os.path.dirname(out_csv))
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for (col, row, idx, sigma_m, hpeak, score) in treetops:
            X, Y = pix_to_xy(transform, col, row)
            w.writerow({
                "x": X, "y": Y, "col": col, "row": row,
                "height": hpeak, "sigma_m": sigma_m, "score": score,
                "crown_r_est_m": 1.8 * sigma_m
            })

def crowns_to_polygons(src, labels, out_path):
    import geopandas as gpd
    from shapely.geometry import shape as shp_shape
    from rasterio.features import shapes as rio_shapes
    arr = np.ascontiguousarray(labels.astype(np.int32))
    geoms, ids = [], []
    for geom, val in rio_shapes(arr, transform=src.transform, connectivity=8):
        if val == 0:
            continue
        geoms.append(shp_shape(geom))
        ids.append(int(val))
    if not geoms:
        log("[poly] no crowns to export.")
        return
    gdf = gpd.GeoDataFrame({"tree_id": ids}, geometry=geoms, crs=src.crs)
    ensure_dir(os.path.dirname(out_path))
    ext = os.path.splitext(out_path)[1].lower()
    if ext == ".gpkg":
        gdf.to_file(out_path, driver="GPKG")
    elif ext in (".geojson", ".json"):
        gdf.to_file(out_path, driver="GeoJSON")
    else:
        out_path = os.path.splitext(out_path)[0] + ".gpkg"
        gdf.to_file(out_path, driver="GPKG")
    log(f"[save] crown polygons -> {out_path}")

def compute_nnd_stats(treetops):
    if len(treetops) < 2:
        return (np.nan, np.nan)
    pts = np.array([[p[0], p[1]] for p in treetops], dtype=np.float32)  # (col,row)
    tree = cKDTree(pts)
    dists, idxs = tree.query(pts, k=2)  # self + NN
    nn = dists[:, 1]
    mean_nn = float(np.nanmean(nn))
    cv_nn = float(np.nanstd(nn, ddof=1) / max(mean_nn, 1e-6))
    return mean_nn, cv_nn

def extract_slim_plot_features(plot_id, recs, treetops, chm_shape, px):
    mean_nn_px, cv_nn = compute_nnd_stats(treetops)
    mean_nnd_m = float(mean_nn_px * px) if np.isfinite(mean_nn_px) else np.nan
    if len(recs) > 0:
        area_sum = float(np.sum([r["area_m2"] for r in recs]))
        cover_ratio = area_sum / (chm_shape[0]*chm_shape[1]*px*px)
        radii = np.array([r["radius_eq_m"] for r in recs], dtype=np.float32)
        mean_radius = float(np.mean(radii)) if radii.size > 0 else np.nan
        cv_radius = float(np.std(radii, ddof=1) / max(np.mean(radii), 1e-6)) if radii.size > 1 else 0.0
        pct_oversize = float(np.mean([r["oversize"] > 0 for r in recs]))
        pct_edge = float(np.mean([r["edge_flag"] > 0 for r in recs]))
        mean_peak_h = float(np.mean([r["peak_h"] for r in recs]))
    else:
        cover_ratio = mean_radius = cv_radius = pct_oversize = pct_edge = mean_peak_h = np.nan
    return {
        "plot_id": plot_id,
        "n_trees": int(len(treetops)),
        "mean_nnd_m": mean_nnd_m,
        "cv_nnd": float(cv_nn) if np.isfinite(cv_nn) else np.nan,
        "cover_ratio": float(cover_ratio),
        "mean_radius_m": float(mean_radius) if np.isfinite(mean_radius) else np.nan,
        "cv_radius": float(cv_radius) if np.isfinite(cv_radius) else np.nan,
        "pct_oversize": float(pct_oversize) if np.isfinite(pct_oversize) else np.nan,
        "pct_edge_trees": float(pct_edge) if np.isfinite(pct_edge) else np.nan,
        "mean_peak_h": float(mean_peak_h) if np.isfinite(mean_peak_h) else np.nan,
    }

# ===================== BATCH (single folder pairing) =====================
# Filenames must contain '_CHM_' or '_LAZ_' and end with .tif/.tiff/.laz
_id_pat = re.compile(r"^(?P<id>.+?)_(?:CHM|LAZ)[^\\\/]*\.(?:tif|tiff|laz)$", re.IGNORECASE)

def collect_pairs_single_dir(data_dir):
    files = glob.glob(os.path.join(data_dir, "*.*"))
    chm_map, laz_map = {}, {}
    for fp in files:
        base = os.path.basename(fp)
        m = _id_pat.match(base)
        if not m:
            continue
        pid = m.group("id")
        ext = os.path.splitext(base)[1].lower()
        if "_chm_" in base.lower() and ext in (".tif", ".tiff"):
            chm_map[pid] = fp
        elif "_laz_" in base.lower() and ext == ".laz":
            laz_map[pid] = fp
    ids = sorted(chm_map.keys())
    pairs = []
    for pid in ids:
        pairs.append((pid, chm_map[pid], laz_map.get(pid, "")))
    return pairs

def process_plot(plot_id, chm_path, laz_path):
    t0 = time.time()
    out_dir = os.path.join(OUT_ROOT, plot_id)
    ensure_dir(out_dir)
    src, chm, px = read_chm(chm_path)

    if FILL_PITS:
        chm = fill_small_pits(chm, px, PIT_RADIUS_M, PIT_DEPTH_M)
        if DEBUG_DUMPS:
            save_tif_like(src, os.path.join(out_dir, "chm_after_pitfill.tif"), chm)

    base_mask = height_mask(chm, MASK_MIN_H, morph_close_px=3)
    can_mask  = canopy_mask_from_height(chm, px, hmin=MASK_MIN_H, min_area_m2=16.0, close_px=3)
    rough_mask= roughness_mask(chm, px, win_m=2.0, std_thr=0.28)
    mask = base_mask & can_mask & rough_mask
    if DEBUG_DUMPS:
        save_tif_like(src, os.path.join(out_dir, "mask_final.tif"), mask.astype("uint8"), dtype="uint8", nodata=0)

    treetops, _, _, chm_vis = detect_multiscale(chm, mask, px)

    # per-plot exports
    export_tops_csv(os.path.join(out_dir, f"{plot_id}_tops.csv"), treetops, src.transform)

    if len(treetops) > 0:
        lbl_tif = os.path.join(out_dir, f"{plot_id}_crowns_labels.tif")
        labels, recs = crowns_via_watershed(chm, treetops, mask, px, src, out_lbl_tif=lbl_tif)
        if len(recs) > 0:
            with open(os.path.join(out_dir, f"{plot_id}_tree_table.csv"), "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(recs[0].keys()))
                w.writeheader()
                w.writerows(recs)
        if DO_POLY and labels.max() > 0:
            try:
                crowns_to_polygons(src, labels, os.path.join(out_dir, f"{plot_id}_crowns.gpkg"))
            except Exception as e:
                log(f"[warn] polygon export skipped: {e}")
    else:
        labels, recs = np.zeros_like(chm, dtype=np.uint32), []
        log("[info] no treetops -> no crowns.")

    feat = extract_slim_plot_features(plot_id, recs, treetops, chm.shape, px)
    feat["_chm_path"] = chm_path
    feat["_laz_path"] = laz_path or ""
    log(f"[done] {plot_id}: trees={feat['n_trees']}, cover={feat['cover_ratio']:.3f}, "
        f"meanR={feat['mean_radius_m'] if feat['mean_radius_m']==feat['mean_radius_m'] else float('nan')}")
    return feat

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Batch ITC detection & delineation on clipped plots where CHM and LAZ "
            "are stored in the same directory."
        )
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Directory containing clipped CHM (.tif/.tiff) and optional LAZ (.laz) files.",
    )
    parser.add_argument(
        "--out-root",
        required=True,
        help="Output root directory for per-plot results and the global master CSV.",
    )
    parser.add_argument(
        "--no-poly",
        action="store_true",
        help="Disable crown polygon (GPKG) export.",
    )
    parser.add_argument(
        "--debug-dumps",
        action="store_true",
        help="Save intermediate rasters (pit-filled CHM, masks, responses).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce console output.",
    )
    return parser.parse_args()

def main():
    global DATA_DIR, OUT_ROOT, DO_POLY, DEBUG_DUMPS, VERBOSE

    args = parse_args()
    DATA_DIR = args.data_dir
    OUT_ROOT = args.out_root
    DO_POLY = not args.no_poly
    DEBUG_DUMPS = args.debug_dumps
    VERBOSE = not args.quiet

    ensure_dir(OUT_ROOT)
    pairs = collect_pairs_single_dir(DATA_DIR)
    if len(pairs) == 0:
        print(f"[error] No CHM found in {DATA_DIR}. Expected filenames like VSNxxxxxx_CHM_*.tif")
        return
    print(f"[batch] Found {len(pairs)} plots in {DATA_DIR}. Starting...")

    master = []
    for (pid, chm_path, laz_path) in pairs:
        try:
            feat = process_plot(pid, chm_path, laz_path)
            master.append(feat)
        except Exception as e:
            print(f"[error] {pid}: {e}")

    master_csv = os.path.join(OUT_ROOT, "master_itc_features.csv")
    cols = SLIM_FEATURES + ["_chm_path","_laz_path"]
    ensure_dir(os.path.dirname(master_csv))
    with open(master_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in master:
            w.writerow({k: r.get(k, "") for k in cols})
    print(f"[save] {master_csv}")
    print("[batch] All done.")

if __name__ == "__main__":
    main()
