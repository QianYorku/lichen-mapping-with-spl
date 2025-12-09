````markdown
# Lichen mapping with Single Photon LiDAR (SPL)
Open-source Python workflow for lichen mapping facilitated by Single-Photon LiDAR (SPL).

## **Overview**
This repository provides a reproducible workflow for lichen mapping facilitated by Single Photon LiDAR (SPL) in boreal mixedwood forests.  
It implements a modular Python pipeline for:

- Coordinate cleaning and consistent UTM projection for field plots  
- Plot-level clipping of SPL canopy height models (CHM) and point clouds (LAZ)  
- Individual Tree Crown (ITC) detection and delineation  
- Plot-level LiDAR feature extraction from CHM and LAZ  
- Lichen presence modeling with multiple classifiers (RF, Logit-EN, SVM-RBF)

The code is designed as an open, self-contained project guide so that researchers and practitioners can adapt the workflow to other regions and forest types.

---

## **Repository structure**

```text
lichen-mapping-with-spl/
├─ src/
│   ├─ verify_and_convert_coords_for_clipping.py
│   ├─ clip_plots_20m.py
│   ├─ itc_delineation_and_plot_features.py
│   ├─ plot_chm_laz_features.py
│   └─ lichen_models_rf_logit_svm.py
│
├─ data/
│   └─ demo_clipped_plots/
│       ├─ DEMO001_CHM_20m.tif
│       ├─ DEMO001_LAZ_20m.laz
│       └─ (optional: a few more tiny CHM/LAZ demo plots)
│
├─ outputs/
│   ├─ itc_delineation/        # ITC crowns & per-plot ITC metrics
│   ├─ plot_level_features/    # CHM/LAZ plot-level features
│   └─ lichen_models/          # model metrics, figures, tables
│
├─ README.md
├─ requirements.txt
├─ .gitignore
└─ LICENSE
````

## **Scripts overview**

All Python scripts are in `src/`:

### **`verify_and_convert_coords_for_clipping.py`**

Cleans and validates the plot coordinate table (CSV).

Uses latitude/longitude to:

* Infer UTM zone and datum, or
* Force a user-specified EPSG code.

Outputs:

* A cleaned coordinate CSV (for clipping)
* An issues report
* Optionally a GeoPackage of plot centers and 20 × 20 m plot squares

---

### **`clip_plots_20m.py`**

Uses the cleaned coordinate CSV and the original SPL CHM/LAZ tiles.

For each plot, it:

* Finds overlapping CHM/LAZ tiles
* Clips a 20 × 20 m window around the plot center
* Diagnoses coverage (single-tile, multi-tile merge, or missing)

Outputs:

* Per-plot CHM and LAZ clips (e.g., `VSNxxxxxx_CHM_20m.tif`, `VSNxxxxxx_LAZ_20m.laz`)
* Coverage status tables and logs

In this repository, you start from already clipped demo plots in `data/demo_clipped_plots/`.

---

### **`itc_delineation_and_plot_features.py`**

Batch ITC detection and crown delineation when CHM and LAZ are in the same folder.

Automatically pairs files such as:

* `VSN601001_CHM_20m.tif` ↔ `VSN601001_LAZ_20m.laz` (LAZ optional)

**ITC detection:**

* Multi-scale Laplacian of Gaussian (LoG) response
* Local height difference (LHD)
* Hessian-based blob filter
* Two-stage non-maximum suppression (NMS)

**Crown delineation:**

* Marker-controlled watershed on a smoothed slope/gradient image

**Outputs per plot:**

* `*_tops.csv` – treetop locations in map coordinates
* `*_crowns_labels.tif` – raster crown labels
* `*_crowns.gpkg` (optional) – crown polygons
* `*_tree_table.csv` – per-tree metrics (height, crown area, radius, edge/oversize flags, etc.)

**Outputs per batch:**

* `master_itc_features.csv` – slim plot-level ITC metrics (tree count, crown cover ratio, mean crown radius, etc.)

---

### **`plot_chm_laz_features.py`**

Extracts plot-level LiDAR features directly from CHM rasters and LAZ point clouds.

Uses filename patterns like:

* `PLOTID_CHM_20m.tif`
* `PLOTID_LAZ_20m.laz`

**CHM features include:**

* Mean, standard deviation, min, max, interquartile range (IQR), skewness, kurtosis
* Canopy cover and gap metrics at height thresholds (e.g., >3 m, >5 m)
* Local roughness / texture using multi-scale window statistics

**LAZ features include:**

* Height distribution statistics and percentiles
* Foliage height diversity (FHD)
* Vertical strata ratios (low / mid / high canopy)
* Return number ratios (first / last / multi)
* Intensity statistics overall and near ground

**Output:**

* `outputs/plot_level_features/lichen_features.csv` (default path)

---

### **`lichen_models_rf_logit_svm.py`**

Builds and evaluates lichen presence models using:

* Random Forest (RF)
* Logistic Regression with Elastic Net (Logit-EN)
* Support Vector Machine with RBF kernel (SVM-RBF)

Assumes the feature table has a binary column `lichen_presence`.

**Workflow:**

* Train/test split with stratification
* Probability calibration (optional)
* Threshold selection via cross-validated Fβ (β = 1.5)
* ROC-AUC, PR-AUC, Brier score, F1, accuracy
* Permutation importance based on Average Precision (AP)
* Feature “share of total” importance and rank percentiles

**Outputs:**

* CSV tables of metrics and permutation importances
* Figures: ROC and PR curves, reliability diagram, confusion matrices
* Plain-text classification reports and a LaTeX-ready metrics table

---

## **Installation**

### **1. Create a Python environment**

It is recommended to use `conda` or `mamba`:

```bash
conda create -n spl_lichen python=3.10
conda activate spl_lichen
```

### **2. Install dependencies**

From the repository root:

```bash
pip install -r requirements.txt
```

If you encounter issues with `rasterio` or `geopandas`, you may install them via `conda` first.

---

## **Demo data**

Small demo clipped plots (e.g., `DEMO001_CHM_20m.tif`, `DEMO001_LAZ_20m.laz`) are provided under:

```text
data/demo_clipped_plots/
```

The full SPL tiles and the original plot coordinate CSV used in the real project are **not** included in this repository.

* The coordinate CSV is only described conceptually in this README.
* Users should replace the paths in the scripts with their own coordinate table and SPL data.

---

## **Example workflow**

### **1. ITC delineation and ITC-based plot features**

```bash
python src/itc_delineation_and_plot_features.py
```

By default, the script assumes:

```python
DATA_DIR = "data/demo_clipped_plots"
OUT_ROOT = "outputs/itc_delineation"
```

Outputs will be written to `outputs/itc_delineation/`.

---

### **2. Plot-level CHM + LAZ features**

```bash
python src/plot_chm_laz_features.py
```

Default paths:

```python
CLIPPED_DIR = Path("data/demo_clipped_plots")
OUTPUT_CSV  = Path("outputs/plot_level_features/lichen_features.csv")
```

You can merge this table with your own lichen ground truth to add a `lichen_presence` column.

---

### **3. Lichen presence modeling**

After adding a `lichen_presence` column to `lichen_features.csv`, run:

```bash
python src/lichen_models_rf_logit_svm.py
```

By default:

```python
file_path = "outputs/plot_level_features/lichen_features.csv"
out_dir   = "outputs/lichen_models"
```

---

## **Data and privacy notes**

* The coordinate CSV and full SPL datasets used in the original project are not shared here due to size and licensing constraints.
* The pipeline is provided so that other users can run the workflow end-to-end on their own data.
* The demo clipped plots are intentionally small and serve only as a technical example of the pipeline.
