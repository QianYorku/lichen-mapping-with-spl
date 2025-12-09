# -*- coding: utf-8 -*-
# verify_and_convert_coords_for_clipping.py
# ------------------------------------------------------------
# Purpose:
# 1) Read a GPS CSV and verify / harmonize coordinates using lat/lon only;
# 2) Two target CRS modes:
#    - AUTO_PER_PLOT: infer UTM zone from longitude and prefer NAD83(CSRS)
#    - FORCE_EPSG: force all plots into a single EPSG (e.g., 2957)
# 3) Quality control (QC): automatic spatial gate (based on distribution),
#    UTM zone consistency, decimal precision, and consistency with any UTM
#    columns in the table.
# 4) Output a cleaned coordinate CSV + an issues report; optionally export a
#    GeoPackage with plot center points and 20 × 20 m squares.
# Dependencies: pandas, numpy, pyproj, (optional) geopandas, shapely
# ------------------------------------------------------------

import math
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import CRS, Transformer


def utm_zone_from_epsg(epsg: int):
    mapping = {32616: 16, 32617: 17, 26916: 16, 26917: 17, 2956: 16, 2957: 17}
    return mapping.get(int(epsg), None)


def datum_from_epsg(epsg: int):
    e = int(epsg)
    if e in (32616, 32617):
        return "WGS84"
    if e in (26916, 26917):
        return "NAD83"
    if e in (2956, 2957):
        return "NAD83(CSRS)"
    return "UNKNOWN"


# ============ CONFIGURATION (generic, no local paths) ============

# Target CRS mode: 'AUTO_PER_PLOT' or 'FORCE_EPSG'
TARGET_MODE = "AUTO_PER_PLOT"  # change to 'FORCE_EPSG' to force a single EPSG below
FORCE_EPSG = 2957              # only used when TARGET_MODE='FORCE_EPSG' (e.g., 2957 NAD83(CSRS)/UTM 17N)

# Automatic gate parameters (based on GPS CSV distribution)
AUTO_P_LO = 0.01   # lower quantile
AUTO_P_HI = 0.99   # upper quantile
AUTO_PAD_DEG = 0.30   # padding around the bounding box in degrees; 0.30° ≈ 33 km * cosφ

# Generic UTM numeric range (for sanity checks)
UTM_E_RANGE = (200_000, 800_000)      # Easting
UTM_N_RANGE = (5_300_000, 5_800_000)  # Northing

# Size of the square polygons to create around each plot center (meters)
SQUARE_SIZE_M = 20.0

# ======================================

SRC_GEO = CRS.from_epsg(4326)  # WGS84 geographic (lon/lat)

# Common candidate CRSs (for Ontario-type data)
CANDIDATES = {
    "EPSG:32616": CRS.from_epsg(32616),  # WGS84 / UTM 16N
    "EPSG:32617": CRS.from_epsg(32617),  # WGS84 / UTM 17N
    "EPSG:26916": CRS.from_epsg(26916),  # NAD83 / UTM 16N
    "EPSG:26917": CRS.from_epsg(26917),  # NAD83 / UTM 17N
    "EPSG:2956": CRS.from_epsg(2956),    # NAD83(CSRS) / UTM 16N
    "EPSG:2957": CRS.from_epsg(2957),    # NAD83(CSRS) / UTM 17N
}


def utm_zone_from_lon(lon):
    return int(math.floor((float(lon) + 180.0) / 6.0) + 1)


def decimals_ok(x, min_dec=6):
    try:
        s = f"{float(x):.10f}"
        return len(s.split(".")[1].rstrip("0")) >= min_dec
    except Exception:
        return False


def project_lonlat(lon, lat, crs):
    tfm = Transformer.from_crs(SRC_GEO, crs, always_xy=True)
    return tfm.transform(float(lon), float(lat))


def auto_bounds_from_gps(
    df,
    lat_col="latitude",
    lon_col="longitude",
    p_lo=0.01,
    p_hi=0.99,
    pad_deg=0.3,
):
    lat = pd.to_numeric(df[lat_col], errors="coerce")
    lon = pd.to_numeric(df[lon_col], errors="coerce")
    lat = lat[np.isfinite(lat)]
    lon = lon[np.isfinite(lon)]
    if len(lat) == 0 or len(lon) == 0:
        # Fallback: very loose bounding box over eastern Canada
        return 45.0, 55.0, -95.0, -70.0, 17
    la_lo, la_hi = np.quantile(lat, [p_lo, p_hi])
    lo_lo, lo_hi = np.quantile(lon, [p_lo, p_hi])
    lat_min = float(la_lo - pad_deg)
    lat_max = float(la_hi + pad_deg)
    lon_min = float(lo_lo - pad_deg)
    lon_max = float(lo_hi + pad_deg)
    # UTM zone mode
    zones = np.floor((lon + 180.0) / 6.0).astype(int) + 1
    zones = zones[(zones >= 1) & (zones <= 60)]
    if len(zones):
        z_mode = int(pd.Series(zones).mode().iloc[0])
        if z_mode not in (16, 17):
            z_mode = 17
    else:
        z_mode = 17
    return lat_min, lat_max, lon_min, lon_max, z_mode


def qc_flags(lat, lon, utmzone_val, e_list, n_list, lat_range, lon_range):
    lat_lo, lat_hi = lat_range
    lon_lo, lon_hi = lon_range
    flags = []
    if pd.isna(lat) or pd.isna(lon):
        return "NA_LATLON"
    if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
        flags.append("RANGE_LL")
    if not (lat_lo <= lat <= lat_hi):
        flags.append("LAT_OUT")
    if not (lon_lo <= lon <= lon_hi):
        flags.append("LON_OUT")
    if lon > 0:
        flags.append("LON_SIGN")
    if not decimals_ok(lat) or not decimals_ok(lon):
        flags.append("LOW_PREC")
    # UTM zone consistency (informational only)
    z_theory = utm_zone_from_lon(lon)
    try:
        z_csv = int(utmzone_val) if pd.notna(utmzone_val) else None
    except Exception:
        z_csv = None
    if z_csv is not None and z_csv not in (16, 17):
        flags.append("UTMZONE_VAL?")
    if z_csv is not None and z_csv != z_theory:
        flags.append(f"UTMZONE_MISMATCH(theory={z_theory})")
    # UTM numeric ranges
    for e, n in zip(e_list, n_list):
        if pd.notna(e) and not (UTM_E_RANGE[0] <= e <= UTM_E_RANGE[1]):
            flags.append("E_RANGE?")
        if pd.notna(n) and not (UTM_N_RANGE[0] <= n <= UTM_N_RANGE[1]):
            flags.append("N_RANGE?")
    return "OK" if not flags else "|".join(sorted(set(flags)))


def best_crs_from_one_utm_pair(lon, lat, e, n):
    """
    When a UTM (E, N) pair exists, enumerate candidate CRSs and choose the one
    that minimizes the difference to the projected lon/lat coordinates.
    """
    if pd.isna(e) or pd.isna(n):
        return None, np.nan
    best, err = None, float("inf")
    for name, crs in CANDIDATES.items():
        try:
            ee, nn = project_lonlat(lon, lat, crs)
            d = math.hypot(ee - float(e), nn - float(n))
            if d < err:
                best, err = name, d
        except Exception:
            continue
    return best, err


def choose_default_crs(lon, utmzone_hint=None):
    """
    Fallback when no reliable UTM columns exist:
    prefer NAD83(CSRS) in the inferred UTM zone.
    """
    z = None
    try:
        if utmzone_hint is not None and not pd.isna(utmzone_hint):
            z = int(utmzone_hint)
    except Exception:
        z = None
    if z not in (16, 17):
        z = utm_zone_from_lon(lon)
    if z == 16:
        return "EPSG:2956", CANDIDATES["EPSG:2956"]
    return "EPSG:2957", CANDIDATES["EPSG:2957"]


def confidence_from_err(min_err, has_utm_any):
    if not has_utm_any:
        return "Low"
    if pd.isna(min_err):
        return "Low"
    if min_err < 3.0:
        return "High"
    if min_err <= 50.0:
        return "Medium"
    return "Low"


def build_square(cx, cy, size_m):
    try:
        from shapely.geometry import Polygon
    except Exception:
        return None
    h = size_m / 2.0
    return Polygon(
        [
            (cx - h, cy - h),
            (cx + h, cy - h),
            (cx + h, cy + h),
            (cx - h, cy + h),
        ]
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Verify and convert plot coordinates for clipping."
    )
    parser.add_argument(
        "--gps-csv",
        type=str,
        required=True,
        help="Input GPS CSV with vsnplotname, latitude, longitude, etc.",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        required=True,
        help="Output CSV with cleaned / harmonized coordinates.",
    )
    parser.add_argument(
        "--out-issues",
        type=str,
        default=None,
        help="Optional issues report CSV (if omitted, the issues report will not be written).",
    )
    parser.add_argument(
        "--export-gpkg",
        action="store_true",
        help="If set, export a GeoPackage with plot centers and 20 m squares.",
    )
    parser.add_argument(
        "--out-gpkg",
        type=str,
        default="gps_centers_20m.gpkg",
        help="Output GeoPackage path (used only when --export-gpkg is set).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    gps_csv = Path(args.gps_csv)
    out_csv = Path(args.out_csv)
    out_issue = Path(args.out_issues) if args.out_issues is not None else None
    export_gpkg = args.export_gpkg
    out_gpkg = Path(args.out_gpkg)

    if not gps_csv.is_file():
        raise FileNotFoundError(gps_csv)

    df = pd.read_csv(gps_csv)

    # Required columns
    req = {"vsnplotname", "latitude", "longitude"}
    if not req.issubset(df.columns):
        raise RuntimeError(
            f"GPS CSV must contain columns: {req}; actual columns: {list(df.columns)}"
        )

    # Automatic bounding box based on lat/lon distribution
    lat_min, lat_max, lon_min, lon_max, zone_mode = auto_bounds_from_gps(
        df, "latitude", "longitude",
        p_lo=AUTO_P_LO, p_hi=AUTO_P_HI, pad_deg=AUTO_PAD_DEG
    )
    print(
        f"[auto-bounds] lat∈[{lat_min:.4f}, {lat_max:.4f}], "
        f"lon∈[{lon_min:.4f}, {lon_max:.4f}] ; UTM zone mode ≈ {zone_mode}"
    )

    # Cast lat/lon to numeric
    df["_lat"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["_lon"] = pd.to_numeric(df["longitude"], errors="coerce")

    # Possible UTM columns (up to three sets)
    e_raw = (
        pd.to_numeric(df.get("easting"), errors="coerce")
        if "easting" in df.columns
        else pd.Series([np.nan] * len(df))
    )
    n_raw = (
        pd.to_numeric(df.get("northing"), errors="coerce")
        if "northing" in df.columns
        else pd.Series([np.nan] * len(df))
    )
    e_asg = (
        pd.to_numeric(df.get("ASSIGNED_EASTING"), errors="coerce")
        if "ASSIGNED_EASTING" in df.columns
        else pd.Series([np.nan] * len(df))
    )
    n_asg = (
        pd.to_numeric(df.get("ASSIGNED_NORTHING"), errors="coerce")
        if "ASSIGNED_NORTHING" in df.columns
        else pd.Series([np.nan] * len(df))
    )
    e_cap = (
        pd.to_numeric(df.get("Easting"), errors="coerce")
        if "Easting" in df.columns
        else pd.Series([np.nan] * len(df))
    )
    n_cap = (
        pd.to_numeric(df.get("Northing"), errors="coerce")
        if "Northing" in df.columns
        else pd.Series([np.nan] * len(df))
    )
    utmzone_series = (
        df.get("UTMZONE")
        if "UTMZONE" in df.columns
        else pd.Series([np.nan] * len(df))
    )

    rows = []
    issues = []

    # If using a forced EPSG, prepare target CRS in advance
    forced_crs = (
        CRS.from_epsg(int(FORCE_EPSG))
        if TARGET_MODE.upper() == "FORCE_EPSG"
        else None
    )
    forced_name = f"EPSG:{FORCE_EPSG}" if forced_crs else None

    # Containers for Geo export
    pt_geoms, sq_geoms = [], []
    gpkg_crs = None  # CRS for GeoPackage export (forced or first successfully inferred CRS)

    for i, r in df.iterrows():
        pid = str(r["vsnplotname"])
        lat = r["_lat"]
        lon = r["_lon"]
        utmzone_val = utmzone_series.iloc[i] if len(utmzone_series) else np.nan

        # QC flags
        qc = qc_flags(
            lat,
            lon,
            utmzone_val,
            [e_raw.iloc[i], e_asg.iloc[i], e_cap.iloc[i]],
            [n_raw.iloc[i], n_asg.iloc[i], n_cap.iloc[i]],
            (lat_min, lat_max),
            (lon_min, lon_max),
        )

        if pd.isna(lat) or pd.isna(lon):
            rows.append(
                {"vsnplotname": pid, "STATUS": "NO_LATLON", "QC_FLAGS": qc}
            )
            issues.append(
                {
                    "vsnplotname": pid,
                    "issue": "NO_LATLON",
                    "detail": "Missing lat/lon",
                }
            )
            pt_geoms.append(None)
            sq_geoms.append(None)
            continue

        # Determine target CRS
        if forced_crs is not None:
            tgt_name, tgt_crs = forced_name, forced_crs
            reason = "forced"
        else:
            # If any UTM columns exist, do a minimum-error match
            pairs = [
                ("RAW", e_raw.iloc[i], n_raw.iloc[i]),
                ("ASSIGNED", e_asg.iloc[i], n_asg.iloc[i]),
                ("CAP", e_cap.iloc[i], n_cap.iloc[i]),
            ]
            choices = []
            for tag, e_val, n_val in pairs:
                name, err = best_crs_from_one_utm_pair(lon, lat, e_val, n_val)
                if name is not None and np.isfinite(err):
                    choices.append((name, err, tag))
            has_utm_any = len(choices) > 0

            if has_utm_any:
                best_name, best_err, best_tag = min(
                    choices, key=lambda x: x[1]
                )
                tgt_name, tgt_crs = best_name, CANDIDATES[best_name]
                reason = f"matched_{best_tag}"
            else:
                tgt_name, tgt_crs = choose_default_crs(
                    lon, utmzone_hint=utmzone_val
                )
                best_err, best_tag = np.nan, "heuristic"
                reason = "heuristic_lon/utmzone"

        # Project center point
        Ex, Ny = project_lonlat(lon, lat, tgt_crs)

        # Differences between projected coordinates and existing UTM columns (in target CRS)
        def diff_pair(E0, N0):
            if pd.isna(E0) or pd.isna(N0):
                return np.nan, np.nan, np.nan
            dE, dN = abs(Ex - float(E0)), abs(Ny - float(N0))
            return dE, dN, math.hypot(dE, dN)

        dE_raw, dN_raw, d_raw = diff_pair(e_raw.iloc[i], n_raw.iloc[i])
        dE_asg, dN_asg, d_asg = diff_pair(e_asg.iloc[i], n_asg.iloc[i])
        dE_cap, dN_cap, d_cap = diff_pair(e_cap.iloc[i], n_cap.iloc[i])

        min_legacy_err = (
            np.nanmin(
                [x for x in [d_raw, d_asg, d_cap] if not pd.isna(x)]
            )
            if any(
                [
                    pd.notna(e_raw.iloc[i]) and pd.notna(n_raw.iloc[i]),
                    pd.notna(e_asg.iloc[i]) and pd.notna(n_asg.iloc[i]),
                    pd.notna(e_cap.iloc[i]) and pd.notna(n_cap.iloc[i]),
                ]
            )
            else np.nan
        )

        conf = confidence_from_err(
            min_legacy_err, has_utm_any=(reason != "heuristic_lon/utmzone")
        )

        epsg_int = int(tgt_crs.to_epsg()) if tgt_crs.to_epsg() is not None else None

        rows.append(
            {
                "vsnplotname": pid,
                "LAT": lat,
                "LON": lon,
                "TARGET_EPSG": f"EPSG:{epsg_int}"
                if epsg_int
                else tgt_crs.to_string(),
                "TARGET_EPSG_INT": epsg_int,  # numeric EPSG
                "TARGET_ZONE": utm_zone_from_epsg(epsg_int)
                if epsg_int
                else None,
                "TARGET_DATUM": datum_from_epsg(epsg_int)
                if epsg_int
                else None,
                "CENTER_E": Ex,
                "CENTER_N": Ny,
                "QC_FLAGS": qc,
                "MIN_LEGACY_ERR_m": min_legacy_err,
                "CONFIDENCE": conf,
                "DECISION_REASON": reason,
                "STATUS": "OK",
                # extra diagnostic columns can be added here if needed
            }
        )

        # Record issues if QC or confidence is not ideal
        if conf == "Low" or (qc != "OK"):
            issues.append(
                {
                    "vsnplotname": pid,
                    "issue": "LOW_CONF_or_QC",
                    "detail": f"QC={qc}; CONF={conf}; reason={reason}; min_legacy_err={min_legacy_err}",
                }
            )

        # Prepare Geo export
        if export_gpkg:
            try:
                from shapely.geometry import Point

                pt_geoms.append(Point(Ex, Ny))
                sq_geoms.append(build_square(Ex, Ny, SQUARE_SIZE_M))
                if gpkg_crs is None:
                    gpkg_crs = (
                        tgt_crs if forced_crs is None else forced_crs
                    )
            except Exception:
                pt_geoms.append(None)
                sq_geoms.append(None)

    # Write cleaned CSV
    out = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False, float_format="%.3f")
    print(f"✅ Clean coordinates CSV -> {out_csv}  ({len(out)} rows)")

    # Write issues report (if requested)
    if issues and out_issue is not None:
        out_issue.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(issues).to_csv(out_issue, index=False)
        print(f"⚠️  Issues report -> {out_issue}")
    elif not issues:
        print("✅ No notable issues flagged by QC/confidence")

    # Optional GeoPackage export (plot centers and 20 × 20 m squares)
    if export_gpkg:
        try:
            import geopandas as gpd

            if gpkg_crs is None:
                # Fallback if we never had a strong CRS choice:
                # use FORCE_EPSG if set, otherwise default to EPSG:2957
                gpkg_crs = (
                    CRS.from_epsg(int(FORCE_EPSG))
                    if TARGET_MODE.upper() == "FORCE_EPSG"
                    else CRS.from_epsg(2957)
                )
            gdf_pt = gpd.GeoDataFrame(out, geometry=pt_geoms, crs=gpkg_crs)
            gdf_sq = gpd.GeoDataFrame(out, geometry=sq_geoms, crs=gpkg_crs)
            out_gpkg.parent.mkdir(parents=True, exist_ok=True)
            gdf_pt.to_file(out_gpkg, layer="plot_centers", driver="GPKG")
            gdf_sq.to_file(
                out_gpkg, layer="plot_20m_squares", driver="GPKG"
            )
            print(
                f"✅ GeoPackage -> {out_gpkg} (layers: plot_centers, plot_20m_squares)"
            )
        except Exception as e:
            print(f"[warn] GPKG export failed: {e}")


if __name__ == "__main__":
    main()
