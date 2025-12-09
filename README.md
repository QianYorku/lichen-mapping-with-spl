# Lichen mapping with Single Photon LiDAR (SPL)
Open-source Python workflow for lichen mapping facilitated by Single-Photon LiDAR (SPL).

This repository provides a reproducible workflow for lichen mapping facilitated by Single Photon LiDAR (SPL) in boreal mixedwood forests.  
It implements a modular Python pipeline for:

- Coordinate cleaning and consistent UTM projection for field plots  
- Plot-level clipping of SPL canopy height models (CHM) and point clouds (LAZ)  
- Individual Tree Crown (ITC) detection and delineation  
- Plot-level LiDAR feature extraction from CHM and LAZ  
- Lichen presence modeling with multiple classifiers (RF, Logit-EN, SVM-RBF)

The code is designed as an open, self-contained project guide so that researchers and practitioners can adapt the workflow to other regions and forest types.

---

## Repository structure

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

## Data availability

Only small demo clipped plots are included under `data/demo_clipped_plots/`.  
The original SPL tiles and field plot coordinate tables (CSV) used in the full project are **not** distributed in this repository.
