# -*- coding: utf-8 -*-
# clip_plots_20m.py
# ------------------------------------------------------------
# Read a unified coordinate CSV (e.g., GPS_clean_for_clipping.csv)
# and clip CHM/LAZ tiles into 20 × 20 m plot windows.
#
# Features:
# - Coverage diagnosis (single tile / union of tiles / incomplete)
# - Use TARGET_EPSG + CENTER_E / CENTER_N as the plot center in a consistent CRS
# - Support plots that may be covered by multiple tiles (e.g., VSN, VSN_1/2/3...)
#
# Dependencies: pandas, numpy, pyproj, rasterio, shapely, laspy, geopandas (optional)
# ------------------------------------------------------------

import os
import re
import math
import warnings
import argparse
from typing import List, Tuple, Optional
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from pyproj import CRS, Transformer
import rasterio
from rasterio.mask import mask  # imported but currently unused; kept for potential extension
from rasterio.merge import merge
from shapely.geometry import box, Polygon

warnings.filterwarnings("ignore")

# ------------- global constants (no local paths here) -------------

SQUARE_SIZE_M = 20.0
BOUNDS_BUFFER_M = 2.0   # buffer (meters) when testing coverage, to tolerate pixel-level offsets
VALID_EXTS = (".tif", ".tiff", ".las", ".laz", ".copc.laz")


# ---------- small utilities ----------

def _norm_key(s: str) -> str:
    """Loose matching key: remove spaces/underscores/hyphens and lowercase."""
    return re.sub(r"[\s\-_]+", "", str(s)).lower()


def _scan_all_files(root: Path, exts=VALID_EXTS) -> List[Path]:
    out: List[Path] = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if fn.startswith("._"):
                continue
            fl = fn.lower()
            if fl.endswith(exts) or fl.endswith(".copc.laz"):
                out.append(Path(dp) / fn)
    return out


def _build_name_index(files: List[Path]) -> dict:
    """
    Build an inverted index keyed by normalized filename stem
    (without extension and without '.copc').
    """
    idx = {}
    for fp in files:
        stem = os.path.splitext(fp.name.lower().replace(".copc", ""))[0]
        k = _norm_key(stem)
        idx.setdefault(k, []).append(fp)
    return idx


def list_plot_files(name_index: dict, plot_name: str, exts=VALID_EXTS) -> List[Path]:
    """
    Within the scanned base directory, return all files whose (normalized)
    basename contains the normalized plot name.
    """
    key = _norm_key(plot_name)
    hits: List[Path] = []
    for k, paths in name_index.items():
        if key in k:
            for p in paths:
                pl = str(p).lower()
                if pl.endswith(exts) or pl.endswith(".copc.laz"):
                    hits.append(p)
    # de-duplicate & sort for reproducibility
    return sorted(set(hits))


def read_crs_and_bounds(path: Path) -> Tuple[Optional[CRS], Optional[Tuple[float, float, float, float]]]:
    """
    Return (CRS, bounds) where bounds = (xmin, ymin, xmax, ymax).
    For rasters, only the header is read (fast).
    For LAS/LAZ, read header metadata.
    """
    try:
        lp = path.suffix.lower()
        full = str(path).lower()
        if full.endswith((".tif", ".tiff")):
            with rasterio.open(path) as src:
                crs = CRS.from_user_input(src.crs) if src.crs else None
                b = src.bounds
                return crs, (b.left, b.bottom, b.right, b.top)
        elif full.endswith((".las", ".laz", ".copc.laz")):
            import laspy
            with laspy.open(path) as r:
                h = r.header
                crs = None
                try:
                    parsed = h.parse_crs()
                    if parsed:
                        crs = CRS.from_user_input(parsed)
                except Exception:
                    pass
                if hasattr(h, "mins") and hasattr(h, "maxs"):
                    xmin, ymin = float(h.mins[0]), float(h.mins[1])
                    xmax, ymax = float(h.maxs[0]), float(h.maxs[1])
                else:
                    xmin = float(getattr(h, "x_min", np.nan))
                    ymin = float(getattr(h, "y_min", np.nan))
                    xmax = float(getattr(h, "x_max", np.nan))
                    ymax = float(getattr(h, "y_max", np.nan))
                if not np.isfinite([xmin, ymin, xmax, ymax]).all():
                    return crs, None
                return crs, (xmin, ymin, xmax, ymax)
    except Exception:
        pass
    return None, None


def proj_points(xy, src: CRS, dst: CRS):
    """
    Project a list of (x, y) from src CRS to dst CRS.
    xy: list of (x, y)
    returns: Nx2 numpy array
    """
    if (
        src == dst
        or (src.to_epsg() is not None and dst.to_epsg() is not None and src.to_epsg() == dst.to_epsg())
    ):
        return np.array(xy, dtype=float)
    tfm = Transformer.from_crs(src, dst, always_xy=True)
    X, Y = tfm.transform([p[0] for p in xy], [p[1] for p in xy])
    return np.column_stack([X, Y])


def square_corners(cx, cy, size_m=SQUARE_SIZE_M):
    h = size_m / 2.0
    return [(cx - h, cy - h), (cx + h, cy - h), (cx + h, cy + h), (cx - h, cy + h)]


def square_samples8(cx, cy, size_m=SQUARE_SIZE_M):
    """
    Return the 4 corners + 4 mid-edge points of the square centered at (cx, cy)
    with width/height = size_m.
    """
    c = square_corners(cx, cy, size_m)
    h = size_m / 2.0
    mids = [(cx, cy - h), (cx + h, cy), (cx, cy + h), (cx - h, cy)]
    return c + mids  # 8 points total


def rect_intersects_bounds(xmin, ymin, xmax, ymax, b):
    return not (
        xmax < b[0] - BOUNDS_BUFFER_M
        or xmin > b[2] + BOUNDS_BUFFER_M
        or ymax < b[1] - BOUNDS_BUFFER_M
        or ymin > b[3] + BOUNDS_BUFFER_M
    )


def inside_bounds(x, y, b):
    return (
        (b[0] - BOUNDS_BUFFER_M <= x <= b[2] + BOUNDS_BUFFER_M)
        and (b[1] - BOUNDS_BUFFER_M <= y <= b[3] + BOUNDS_BUFFER_M)
    )


# ---------- coverage diagnosis ----------

def diagnose_coverage(
    plot_crs: CRS,
    cx: float,
    cy: float,
    file_infos: List[Tuple[Path, CRS, Tuple[float, float, float, float]]],
):
    """
    Diagnose coverage for a 20 × 20 m square around (cx, cy) in plot_crs.

    Returns: (cover_mode, hit_single, hit_union, missing_idx)
      - cover_mode: 'single' / 'union' / 'none'
      - hit_single: list of files that individually cover all 4 corners
      - hit_union : list of files that together cover all 8 sample points
      - missing_idx: indices (0–3 corners, 4–7 mid-edge) of any uncovered points
    """
    pts = square_samples8(cx, cy, SQUARE_SIZE_M)
    corners = pts[:4]
    covered_single = False
    hit_single: List[Path] = []
    anycover_files = set()
    covered_flags = [False] * 8

    for path, fcrs, b in file_infos:
        if fcrs is None or b is None:
            continue
        # project sample points to this file's CRS
        pts_f = proj_points(pts, plot_crs, fcrs)
        corners_f = pts_f[:4, :]
        # single-tile coverage: all 4 corners inside
        all_in = all(inside_bounds(float(x), float(y), b) for (x, y) in corners_f)
        if all_in:
            covered_single = True
            hit_single.append(path)
        # any point inside, used for union coverage
        hit_any = False
        for i, (x, y) in enumerate(pts_f):
            if inside_bounds(float(x), float(y), b):
                covered_flags[i] = True
                hit_any = True
        if hit_any:
            anycover_files.add(path)

    if covered_single:
        return "single", sorted(hit_single), sorted(anycover_files), []

    if all(covered_flags):
        return "union", [], sorted(anycover_files), []

    missing_idx = [i for i, v in enumerate(covered_flags) if not v]
    return "none", [], sorted(anycover_files), missing_idx


# ---------- clipping ----------

def clip_chm_one(path: Path, boundary_plot_crs: Polygon, plot_crs: CRS, out_path: Path) -> Optional[Path]:
    """
    Pixel-aligned clipping:
    - use the plot center in the CHM CRS
    - clip an N×N window where N ≈ SQUARE_SIZE_M / pixel_size (rounded to an even integer)
    """
    try:
        with rasterio.open(path) as src:
            fcrs = CRS.from_user_input(src.crs) if src.crs else None
            if fcrs is None:
                return None

            # 1) Project the plot center from unified CRS to CHM CRS
            cx, cy = np.array(boundary_plot_crs.centroid.coords[0], dtype=float)
            tfm = Transformer.from_crs(plot_crs, fcrs, always_xy=True)
            cx_f, cy_f = tfm.transform(cx, cy)

            # 2) determine window size in pixels (N×N), try to approximate 20 m
            px_x, px_y = abs(src.transform.a), abs(src.transform.e)
            if px_x <= 0 or px_y <= 0:
                return None
            n_x = int(round(SQUARE_SIZE_M / px_x))
            n_y = int(round(SQUARE_SIZE_M / px_y))
            # enforce even size for symmetry
            if n_x % 2 == 1:
                n_x += 1
            if n_y % 2 == 1:
                n_y += 1
            n_x = max(2, n_x)
            n_y = max(2, n_y)

            # 3) center → (col, row), then build a pixel-aligned window
            c, r = src.index(cx_f, cy_f)  # (col, row)
            from rasterio.windows import Window

            col_off = int(c - n_x // 2)
            row_off = int(r - n_y // 2)
            win = Window(col_off=col_off, row_off=row_off, width=n_x, height=n_y)

            # 4) intersect with dataset bounds
            win = win.intersection(Window(0, 0, src.width, src.height))
            if win.width <= 0 or win.height <= 0:
                return None

            # 5) read & write
            data = src.read(window=win)
            if data.size == 0:
                return None
            meta = src.meta.copy()
            meta.update(
                {
                    "height": int(win.height),
                    "width": int(win.width),
                    "transform": rasterio.windows.transform(win, src.transform),
                }
            )
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with rasterio.open(out_path, "w", **meta) as dst:
                dst.write(data)
            return out_path
    except Exception:
        return None


def clip_laz_many(
    paths: List[Path],
    boundary_plot_crs: Polygon,
    plot_crs: CRS,
    out_path: Path,
) -> Optional[Path]:
    """
    Clip multiple LAS/LAZ files and merge them into a single output point cloud.

    Notes:
    - We avoid writing COPC; a standard LAS/LAZ is written.
    - The point_format and dimensions of the first file are used as the template.
    - Subsequent files whose point_format.id differ from the first file are skipped
      (with a warning).
    - If LAZ compression is not available, we fall back to writing LAS.
    """
    import laspy
    from pyproj import CRS as PJCRS

    # bounding rectangle in plot CRS
    bb_minx, bb_miny, bb_maxx, bb_maxy = boundary_plot_crs.bounds

    target_header = None
    target_pf_id = None
    target_dims = None
    target_epsg = None

    collected = {}  # dim -> [chunks...]
    total_points = 0
    skipped_formats = set()

    for path in paths:
        fcrs, b = read_crs_and_bounds(path)
        if (fcrs is None) or (b is None):
            continue

        # project bounding box into this file's CRS for a quick intersection test
        bb = proj_points([(bb_minx, bb_miny), (bb_maxx, bb_maxy)], plot_crs, fcrs)
        fx0, fy0 = float(min(bb[0, 0], bb[1, 0])), float(min(bb[0, 1], bb[1, 1]))
        fx1, fy1 = float(max(bb[0, 0], bb[1, 0])), float(max(bb[0, 1], bb[1, 1]))
        if not rect_intersects_bounds(fx0, fy0, fx1, fy1, b):
            continue

        try:
            with laspy.open(path) as reader:
                hdr = reader.header

                # initialize template from the first file
                if target_header is None:
                    target_header = hdr
                    target_pf_id = hdr.point_format.id
                    target_dims = list(hdr.point_format.dimension_names)
                    # try to extract EPSG from header CRS
                    try:
                        crs_obj = hdr.parse_crs()
                        if crs_obj is not None:
                            e = PJCRS.from_user_input(crs_obj).to_epsg()
                            if e:
                                target_epsg = int(e)
                    except Exception:
                        pass

                # skip tiles with different point_format
                if hdr.point_format.id != target_pf_id:
                    skipped_formats.add((path.name, hdr.point_format.id))
                    continue

                # iterate over points in chunks (if available)
                def iter_points():
                    if hasattr(reader, "chunk_iterator"):
                        for pts in reader.chunk_iterator(1_000_000):
                            yield pts
                    else:
                        yield reader.read()

                for pts in iter_points():
                    x = pts.x
                    y = pts.y
                    inside = (x >= fx0) & (x <= fx1) & (y >= fy0) & (y <= fy1)
                    if not np.any(inside):
                        continue

                    n_in = int(np.count_nonzero(inside))
                    # collect only the dimensions present in the template
                    for dim in target_dims:
                        arr = getattr(pts, dim, None)
                        if arr is None:
                            continue
                        collected.setdefault(dim, []).append(arr[inside])
                    total_points += n_in

        except Exception as e:
            print(f"⚠️ LAZ read error: {path} -> {e}")
            continue

    if total_points == 0 or not collected:
        return None

    # assemble the output point cloud
    try:
        from laspy import LasHeader, LasData
    except Exception as e:
        print(f"⚠️ laspy import error: {e}")
        return None

    new_hdr = LasHeader(point_format=target_header.point_format, version="1.4")
    try:
        new_hdr.scales = target_header.scales
        new_hdr.offsets = target_header.offsets
    except Exception:
        pass
    # write CRS if possible
    try:
        if target_epsg:
            from laspy.vlrs.geotiff import GeoKeyDirectoryVlr
            new_hdr.vlrs.add(GeoKeyDirectoryVlr.from_epsg(int(target_epsg)))
    except Exception:
        pass

    out = LasData(new_hdr)

    # concatenate arrays per dimension
    dims_order = list(target_dims)  # e.g., ["X", "Y", "Z", "intensity", ...]
    arrays = {}
    for dim in dims_order:
        if dim not in collected:
            continue
        arrays[dim] = np.concatenate(collected[dim], axis=0)

    # require at least X/Y/Z
    for base in ("X", "Y", "Z"):
        if base not in arrays:
            print("⚠️ Missing core dimension:", base)
            return None

    n = arrays["X"].shape[0]
    # set X/Y/Z first
    setattr(out, "X", arrays["X"])
    setattr(out, "Y", arrays["Y"])
    setattr(out, "Z", arrays["Z"])
    # assign other dimensions if length matches
    for dim in dims_order:
        if dim in ("X", "Y", "Z"):
            continue
        arr = arrays.get(dim, None)
        if arr is not None and arr.shape[0] == n:
            setattr(out, dim, arr)

    # write output: try LAZ first, then fall back to LAS
    save_path = out_path
    save_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        out.write(str(save_path))   # requires lazrs/laszip for LAZ
    except Exception:
        save_path = out_path.with_suffix(".las")
        try:
            out.write(str(save_path))
        except Exception as e2:
            print(f"⚠️ LAZ/LAS write error: {e2}")
            return None

    if skipped_formats:
        print(
            "ℹ️ Skipped tiles whose point_format differs from the first file: ",
            "; ".join([f"{nm}(pf={pf})" for nm, pf in skipped_formats]),
        )
    return save_path


# ---------- main pipeline ----------

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Clip CHM and LAS/LAZ tiles into 20 × 20 m plots "
            "using unified plot coordinates."
        )
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        required=True,
        help="Root directory containing CHM and LAS/LAZ tiles.",
    )
    parser.add_argument(
        "--coord-csv",
        type=str,
        required=True,
        help=(
            "CSV with unified coordinates (e.g., GPS_clean_for_clipping.csv), "
            "containing vsnplotname, TARGET_EPSG or TARGET_EPSG_INT, CENTER_E, CENTER_N."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to write clipped CHM and LAS/LAZ plots.",
    )
    parser.add_argument(
        "--status-csv",
        type=str,
        default=None,
        help="Optional path for clipping status CSV (default: <output-dir>/clipping_status.csv).",
    )
    parser.add_argument(
        "--log-csv",
        type=str,
        default=None,
        help="Optional path for clipping log CSV (default: <output-dir>/clipping_log.csv).",
    )
    parser.add_argument(
        "--need-csv",
        type=str,
        default=None,
        help=(
            "Optional path for listing plots that need additional coverage "
            "(default: <output-dir>/needs_download.csv)."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    base_dir = Path(args.base_dir)
    coord_csv = Path(args.coord_csv)
    output_dir = Path(args.output_dir)
    status_csv = Path(args.status_csv) if args.status_csv else output_dir / "clipping_status.csv"
    log_csv = Path(args.log_csv) if args.log_csv else output_dir / "clipping_log.csv"
    need_csv = Path(args.need_csv) if args.need_csv else output_dir / "needs_download.csv"

    output_dir.mkdir(parents=True, exist_ok=True)

    if not coord_csv.is_file():
        raise FileNotFoundError(coord_csv)
    if not base_dir.is_dir():
        raise NotADirectoryError(base_dir)

    df = pd.read_csv(coord_csv)

    # case-insensitive column mapping
    colmap = {c.lower(): c for c in df.columns}

    def col(name: str) -> str:
        return colmap.get(name.lower(), name)

    has_any_epsg = (col("TARGET_EPSG") in df.columns) or (col("TARGET_EPSG_INT") in df.columns)
    has_center = (col("CENTER_E") in df.columns) and (col("CENTER_N") in df.columns)
    if not ((col("vsnplotname") in df.columns) and has_any_epsg and has_center):
        raise RuntimeError(
            "Coordinate CSV must contain: vsnplotname, TARGET_EPSG (or TARGET_EPSG_INT), CENTER_E, CENTER_N."
        )

    # build file index under base_dir (scan once)
    all_files = _scan_all_files(base_dir, exts=VALID_EXTS)
    name_index = _build_name_index(all_files)

    status_rows = []
    log_rows = []

    for _, r in df.iterrows():
        pid = str(r[col("vsnplotname")]).strip().upper()

        # determine plot CRS (EPSG preferred)
        epsg = None
        if col("TARGET_EPSG_INT") in df.columns and pd.notna(r.get(col("TARGET_EPSG_INT"))):
            epsg = int(r[col("TARGET_EPSG_INT")])
        elif col("TARGET_EPSG") in df.columns and pd.notna(r.get(col("TARGET_EPSG"))):
            te = str(r[col("TARGET_EPSG")])
            if te.upper().startswith("EPSG:"):
                epsg = int(te.split(":")[1])

        if epsg is None:
            # fallback: infer UTM zone from LON/LAT if available
            has_lon = col("LON") in df.columns and pd.notna(r.get(col("LON")))
            has_lat = col("LAT") in df.columns and pd.notna(r.get(col("LAT")))
            if has_lon and has_lat:
                lon = float(r[col("LON")])
                lat = float(r[col("LAT")])
                zone = int(math.floor((lon + 180) / 6) + 1)
                north = lat >= 0
                plot_crs = CRS.from_dict(
                    {"proj": "utm", "zone": zone, "datum": "WGS84", "south": (not north)}
                )
            else:
                raise RuntimeError(
                    f"{pid}: Neither TARGET_EPSG nor LON/LAT columns are available. "
                    "Cannot determine a projected CRS."
                )
        else:
            plot_crs = CRS.from_epsg(epsg)

        cx, cy = float(r[col("CENTER_E")]), float(r[col("CENTER_N")])
        boundary_plot = box(cx - 10.0, cy - 10.0, cx + 10.0, cy + 10.0)  # 20 × 20 m

        # find candidate files
        files = list_plot_files(name_index, pid)
        chm_files = [f for f in files if str(f).lower().endswith((".tif", ".tiff"))]
        laz_files = [f for f in files if str(f).lower().endswith((".las", ".laz", ".copc.laz"))]

        # read CRS + bounds
        chm_infos = []
        for f in chm_files:
            fcrs, b = read_crs_and_bounds(f)
            if (fcrs is not None) and (b is not None):
                chm_infos.append((f, fcrs, b))
        laz_infos = []
        for f in laz_files:
            fcrs, b = read_crs_and_bounds(f)
            # if LAS/LAZ has no CRS, try to adopt CRS from a sibling TIF in the same folder
            if fcrs is None or b is None:
                d = f.parent
                for fn in os.listdir(d):
                    if fn.lower().endswith((".tif", ".tiff")):
                        with rasterio.open(d / fn) as src:
                            if src.crs:
                                fcrs = CRS.from_user_input(src.crs)
                                break
            if (fcrs is not None) and (b is not None):
                laz_infos.append((f, fcrs, b))

        # coverage diagnosis for CHM and LAZ separately
        chm_mode, chm_single, chm_union, chm_missing = diagnose_coverage(plot_crs, cx, cy, chm_infos)
        laz_mode, laz_single, laz_union, laz_missing = diagnose_coverage(plot_crs, cx, cy, laz_infos)

        # ---- CHM clipping ----
        chm_status = "Missing"
        chm_used: List[Path] = []
        if chm_infos:
            to_use = chm_single if chm_mode == "single" else chm_union
            # if union is empty but we still have files, try all candidate tiles that intersect
            if not to_use:
                to_use = [p for (p, _, _) in chm_infos]
            clips = []
            with TemporaryDirectory() as tmpdir:
                tmpdir_path = Path(tmpdir)
                for i, fp in enumerate(to_use):
                    out_tmp = tmpdir_path / f"chm_{i}.tif"
                    saved = clip_chm_one(fp, boundary_plot, plot_crs, out_tmp)
                    if saved is not None:
                        clips.append(saved)
                        chm_used.append(fp)
                if clips:
                    srcs = [rasterio.open(str(c)) for c in clips]
                    mosaic, out_transform = merge(srcs)
                    meta = srcs[0].meta.copy()
                    for s in srcs:
                        s.close()
                    meta.update(
                        {
                            "height": mosaic.shape[1],
                            "width": mosaic.shape[2],
                            "transform": out_transform,
                        }
                    )
                    chm_out = output_dir / f"{pid}_CHM_20m.tif"
                    with rasterio.open(chm_out, "w", **meta) as dst:
                        dst.write(mosaic)
                    chm_status = "Single" if len(chm_used) == 1 else "Merged"
                else:
                    chm_status = "NoOverlap"

        # ---- LAZ clipping ----
        laz_status = "Missing"
        laz_used: List[Path] = []
        if laz_infos:
            to_use = laz_single if laz_mode == "single" else laz_union
            if not to_use:
                to_use = [p for (p, _, _) in laz_infos]
            laz_out = output_dir / f"{pid}_LAZ_20m.laz"
            saved = clip_laz_many(to_use, boundary_plot, plot_crs, laz_out)
            if saved is not None:
                laz_status = "Single" if len(to_use) == 1 else "Merged"
                laz_used = to_use
            else:
                laz_status = "NoOverlap"

        # ---- report & logging ----
        need_flag = False
        need_reason = []
        if (chm_mode == "none") or (chm_status in ("Missing", "NoOverlap")):
            need_flag = True
            need_reason.append("CHM_INCOMPLETE")
        if (laz_mode == "none") or (laz_status in ("Missing", "NoOverlap")):
            need_flag = True
            need_reason.append("LAZ_INCOMPLETE")

        status_rows.append(
            {
                "PlotID": pid,
                "CHM_MODE": chm_mode,
                "CHM_STATUS": chm_status,
                "LAZ_MODE": laz_mode,
                "LAZ_STATUS": laz_status,
                "CHM_MISSING_IDX": ",".join(map(str, chm_missing)) if chm_missing else "",
                "LAZ_MISSING_IDX": ",".join(map(str, laz_missing)) if laz_missing else "",
                "NEED_DOWNLOAD": "YES" if need_flag else "NO",
                "REASON": "|".join(need_reason),
            }
        )

        if chm_used:
            log_rows.append(
                {"PlotID": pid, "Type": "CHM", "UsedFiles": ";".join(str(p) for p in chm_used)}
            )
        if laz_used:
            log_rows.append(
                {"PlotID": pid, "Type": "LAZ", "UsedFiles": ";".join(str(p) for p in laz_used)}
            )

        print(
            f"[{pid}] CHM:{chm_status}/{chm_mode}  "
            f"LAZ:{laz_status}/{laz_mode}  "
            f"{'-> NEED_DOWNLOAD' if need_flag else ''}"
        )

    # write logs
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(status_rows).to_csv(status_csv, index=False)
    pd.DataFrame(log_rows).to_csv(log_csv, index=False)
    need_df = pd.DataFrame([r for r in status_rows if r["NEED_DOWNLOAD"] == "YES"])
    if not need_df.empty:
        need_df.to_csv(need_csv, index=False)
        print(f"\n❗ Plots requiring additional data -> {need_csv}")
    print(f"📋 Clipping status -> {status_csv}\n📋 Source tiles used -> {log_csv}")


if __name__ == "__main__":
    main()
