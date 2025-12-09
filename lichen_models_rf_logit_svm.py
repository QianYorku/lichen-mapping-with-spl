# ===========================================
# Lichen presence — RF + Logit-EN + SVM-RBF (NO feature selection)
# - Train/test split, calibration, F_beta threshold scan (β=1.5)
# - Metrics tables + publication-quality figures
# - Permutation importance (AP) with share-of-total (%) for EACH model
# ===========================================

import os, json, warnings, re
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, roc_auc_score, average_precision_score,
    brier_score_loss, RocCurveDisplay, PrecisionRecallDisplay, ConfusionMatrixDisplay
)
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay
from sklearn.inspection import permutation_importance
from sklearn.base import clone

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


"""
lichen_models_rf_logit_svm.py

Purpose
-------
Train and evaluate three probabilistic classifiers for lichen presence:
    - Random Forest
    - Logistic Regression (Elastic Net)
    - SVM with RBF kernel

Input
-----
- A plot-level feature table with a binary target column 'lichen_presence'.
  In the public repo we expect the file:
      ./demo_clipped_plots/lichen_features.csv

Output
------
Under `out_dir` (see user settings below), the script produces:
    - metrics_table_all_models.csv / .tex
    - permutation importance CSVs (per model + combined)
    - ROC / PR curves, reliability diagram, confusion matrices
    - RF classification reports at threshold=0.5 and CV-optimized threshold
"""

# ---------------- User settings ----------------
# For the public/demo version, keep these paths relative
file_path = "./demo_clipped_plots/lichen_features.csv"          # input feature table
out_dir   = "./demo_clipped_plots/lichen_model_outputs"         # output root directory

CALIBRATE = True
F_BETA    = 1.5
TEST_SIZE = 0.20
RSEED     = 42
# Number of permutation repetitions for importance (larger -> more stable)
N_REPEATS_PI = 200

# ---------------- Boilerplate ----------------
warnings.filterwarnings("ignore")
np.random.seed(RSEED)
mpl.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300, "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.size": 10, "axes.labelsize": 10, "axes.titlesize": 11,
    "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9, "axes.linewidth": 0.8,
})
PAPER_W, SMALL_W = 6.5, 3.2

fig_dir = os.path.join(out_dir, "figures")
tab_dir = os.path.join(out_dir, "tables")
os.makedirs(fig_dir, exist_ok=True)
os.makedirs(tab_dir, exist_ok=True)


def savefig_all(path_wo_ext: str):
    """Save current Matplotlib figure as PNG + PDF and close it."""
    plt.tight_layout()
    plt.savefig(path_wo_ext + ".png", dpi=300, bbox_inches="tight")
    plt.savefig(path_wo_ext + ".pdf", bbox_inches="tight")
    plt.close()


# ---------------- Load & prepare ----------------
df = pd.read_csv(file_path)
target_col = "lichen_presence"
if target_col not in df.columns:
    raise ValueError(f"Cannot find target column '{target_col}' in CSV.")

non_feature_cols = [c for c in ["PlotID", "plot_id", target_col] if c in df.columns]
X_all = (
    df.drop(columns=non_feature_cols, errors="ignore")
      .select_dtypes(include=[np.number])
      .copy()
)
y = df[target_col].astype(int).copy()

X_train_df, X_test_df, y_train, y_test = train_test_split(
    X_all, y, test_size=TEST_SIZE, stratify=y, random_state=RSEED
)

pd.Series(X_train_df.index, name="train_index").to_csv(
    os.path.join(tab_dir, "train_indices.csv"), index=False
)
pd.Series(X_test_df.index, name="test_index").to_csv(
    os.path.join(tab_dir, "test_indices.csv"), index=False
)

# Missing-value imputation for all models (trees / linear / SVM).
# Linear/SVM then apply standardization inside their pipelines.
imputer = SimpleImputer(strategy="median")
X_train_imp = imputer.fit_transform(X_train_df)
X_test_imp  = imputer.transform(X_test_df)

# ---------------- Models (no feature selection) ----------------
# 1) Random Forest
rf_base = RandomForestClassifier(
    n_estimators=600,
    max_depth=12,
    max_features="sqrt",
    min_samples_leaf=2,
    bootstrap=True,
    class_weight=None,
    random_state=RSEED,
    n_jobs=-1,
)
rf_model = (
    CalibratedClassifierCV(rf_base, method="isotonic", cv=5)
    if CALIBRATE else rf_base
)

# 2) Logistic Regression (Elastic Net) + scaling
logit_pipe = Pipeline(
    [
        ("scaler", StandardScaler()),
        (
            "clf",
            LogisticRegression(
                penalty="elasticnet",
                solver="saga",
                l1_ratio=0.5,
                C=1.0,
                max_iter=5000,
                class_weight="balanced",
                random_state=RSEED,
            ),
        ),
    ]
)
logit_model = (
    CalibratedClassifierCV(logit_pipe, method="sigmoid", cv=5)
    if CALIBRATE else logit_pipe
)

# 3) SVM (RBF) + scaling
svm_pipe = Pipeline(
    [
        ("scaler", StandardScaler()),
        (
            "clf",
            SVC(
                kernel="rbf",
                probability=True,
                C=1.0,
                gamma="scale",
                class_weight="balanced",
                random_state=RSEED,
            ),
        ),
    ]
)
svm_model = (
    CalibratedClassifierCV(svm_pipe, method="sigmoid", cv=5)
    if CALIBRATE else svm_pipe
)

models = {
    "RF": rf_model,
    "LogitEN": logit_model,
    "SVM_RBF": svm_model,
}


# ---------------- Helpers ----------------
def fbeta_at_threshold(y_true, proba, t, beta):
    """Compute F_beta at a given decision threshold t."""
    pred = (proba >= t).astype(int)
    tp = np.sum((pred == 1) & (y_true == 1))
    fp = np.sum((pred == 1) & (y_true == 0))
    fn = np.sum((pred == 0) & (y_true == 1))

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    b2 = beta**2
    if (prec + rec) == 0 or (b2 * prec + rec) == 0:
        return 0.0
    return (1 + b2) * (prec * rec) / (b2 * prec + rec)


def pick_threshold_via_cv(model, X_tr, y_tr, beta, ths=None, n_splits=5):
    """
    Scan thresholds in [0.01, 0.99] and pick the one that maximizes F_beta,
    averaged via StratifiedKFold on the training set.
    """
    ths = np.linspace(0.01, 0.99, 199) if ths is None else ths
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RSEED)
    best_ts = []
    for tr_idx, va_idx in cv.split(X_tr, y_tr):
        m = clone(model).fit(X_tr[tr_idx], y_tr.iloc[tr_idx])
        proba = m.predict_proba(X_tr[va_idx])[:, 1]
        fbs = [
            fbeta_at_threshold(y_tr.iloc[va_idx].values, proba, t, beta)
            for t in ths
        ]
        best_ts.append(float(ths[int(np.argmax(fbs))]))
    return float(np.median(best_ts))


def metrics_block(y_true, proba, y_pred, label, threshold_value=None):
    """Collect a compact block of metrics for one model setting."""
    out = {"label": label, "threshold": threshold_value}
    out["roc_auc"] = roc_auc_score(y_true, proba)
    out["pr_auc"] = average_precision_score(y_true, proba)
    out["brier"] = brier_score_loss(y_true, proba)
    if y_pred is not None:
        out["accuracy"] = accuracy_score(y_true, y_pred)
        out["f1"] = f1_score(y_true, y_pred, pos_label=1)
    return out


def compute_perm_importance_with_percentages(
    model, X_eval, y_eval, feature_names, model_name, outdir_csv, outdir_fig
):
    """
    Permutation importance using AP, plus:
        - share_of_total (% of total positive importance)
        - relative_drop (% change relative to baseline AP)
        - rank percentile (0–100, 100 = most important)
    """
    # Baseline AP on the evaluation set
    proba = model.predict_proba(X_eval)[:, 1]
    base_ap = average_precision_score(y_eval, proba)

    perm = permutation_importance(
        model,
        X_eval,
        y_eval,
        n_repeats=N_REPEATS_PI,
        random_state=RSEED,
        n_jobs=-1,
        scoring="average_precision",
    )
    imp = pd.DataFrame(
        {
            "model": model_name,
            "feature": feature_names,
            "perm_importance_mean": perm.importances_mean,
            "perm_importance_std": perm.importances_std,
        }
    )

    # Share-of-total, relative drop, rank percentile
    imp["imp_pos"] = imp["perm_importance_mean"].clip(lower=0.0)
    total = imp["imp_pos"].sum()
    imp["share_percent"] = 0.0 if total <= 0 else 100.0 * imp["imp_pos"] / total
    imp["rel_drop_percent"] = (
        0.0 if base_ap <= 0 else 100.0 * imp["perm_importance_mean"] / base_ap
    )

    imp = imp.sort_values("perm_importance_mean", ascending=False).reset_index(
        drop=True
    )
    n = len(imp)
    imp["rank_percentile"] = 100.0 if n <= 1 else (1.0 - imp.index / (n - 1)) * 100.0

    # Export CSV
    cols = [
        "model",
        "feature",
        "perm_importance_mean",
        "perm_importance_std",
        "share_percent",
        "rel_drop_percent",
        "rank_percentile",
    ]
    imp[cols].to_csv(
        os.path.join(outdir_csv, f"perm_importance_{model_name}_percent.csv"),
        index=False,
    )

    # Plot top-20 share-of-total
    top = imp.head(20).iloc[::-1]
    plt.figure(figsize=(PAPER_W, 0.28 * len(top) + 0.6))
    plt.barh(top["feature"], top["share_percent"])
    plt.xlabel("Permutation importance — share of total (%)")
    plt.title(f"Top-20 features (Permutation, % of total) — {model_name}")
    savefig_all(os.path.join(outdir_fig, f"fig_perm_share_{model_name}_top20"))
    return imp, base_ap


# ---------------- Train, calibrate, evaluate, importance ----------------
all_metrics = []
probas_for_curves = {}
best_t_by_model = {}
importances_all = []

for name, model in models.items():
    # Fit model on the imputed training matrix (no feature selection)
    m = clone(model).fit(X_train_imp, y_train)

    # Threshold selection via 5-fold CV on training data using F_BETA
    best_t = pick_threshold_via_cv(m, X_train_imp, y_train, F_BETA)
    best_t_by_model[name] = best_t

    # Test-set probabilities and predictions
    proba = m.predict_proba(X_test_imp)[:, 1]
    y_pred_050 = (proba >= 0.50).astype(int)
    y_pred_opt = (proba >= best_t).astype(int)

    # Metrics
    all_metrics += [
        metrics_block(y_test, proba, y_pred_050, f"{name}@0.50", 0.50),
        metrics_block(y_test, proba, y_pred_opt, f"{name}@cv_opt", best_t),
        metrics_block(y_test, proba, None, f"{name}_proba_only", None),
    ]
    probas_for_curves[name] = proba

    # Permutation importance (AP) with share-of-total / relative drop / rank percentile
    imp_df, base_ap = compute_perm_importance_with_percentages(
        m, X_test_imp, y_test, X_all.columns.tolist(), name, tab_dir, fig_dir
    )
    imp_df["baseline_ap"] = base_ap
    importances_all.append(imp_df)

# Metrics table
metrics_df = pd.DataFrame(all_metrics)
metrics_df.to_csv(
    os.path.join(tab_dir, "metrics_table_all_models.csv"), index=False
)

# Concatenate permutation importances
imp_concat = pd.concat(importances_all, ignore_index=True)
imp_concat.to_csv(
    os.path.join(tab_dir, "perm_importance_all_models_percent.csv"), index=False
)

# ---------------- Curves & confusion matrices ----------------
# ROC curves (all models in one panel)
fig, ax = plt.subplots(figsize=(3.8, 3.1))
for name, proba in probas_for_curves.items():
    RocCurveDisplay.from_predictions(
        y_test,
        proba,
        ax=ax,
        name=f"{name} (AUC={roc_auc_score(y_test, proba):.3f})",
    )
ax.plot([0, 1], [0, 1], "--", lw=1, color="gray")
ax.set_title("ROC — Model comparison")
savefig_all(os.path.join(fig_dir, "fig_ROC_models"))

# PR curves (all models in one panel)
fig, ax = plt.subplots(figsize=(3.8, 3.1))
for name, proba in probas_for_curves.items():
    PrecisionRecallDisplay.from_predictions(
        y_test,
        proba,
        ax=ax,
        name=f"{name} (AP={average_precision_score(y_test, proba):.3f})",
    )
ax.set_title("PR — Model comparison")
savefig_all(os.path.join(fig_dir, "fig_PR_models"))

# Reliability diagram (example: RF; can duplicate for each model if desired)
if "RF" in probas_for_curves:
    fig, ax = plt.subplots(figsize=(SMALL_W, SMALL_W * 0.82))
    CalibrationDisplay.from_predictions(
        y_test,
        probas_for_curves["RF"],
        n_bins=8,
        ax=ax,
        name="RF prob",
    )
    ax.set_title("Reliability Diagram — RF")
    savefig_all(os.path.join(fig_dir, "fig_reliability_RF"))

# Confusion matrices (each model at its own CV-optimized threshold)
for name, proba in probas_for_curves.items():
    t = best_t_by_model[name]
    pred_opt = (proba >= t).astype(int)
    cm = confusion_matrix(y_test, pred_opt, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(SMALL_W, SMALL_W * 0.82))
    ConfusionMatrixDisplay(
        cm, display_labels=[0, 1]
    ).plot(ax=ax, cmap="Greens", colorbar=False, values_format="d")
    ax.set_title(
        f"Confusion Matrix — {name} (threshold={t:.3f}, Fβ={F_BETA})"
    )
    plt.tight_layout()
    savefig_all(os.path.join(fig_dir, f"fig_confmat_{name}_cvopt"))

# ---------------- Save reports ----------------
# Text reports for RF (at 0.50 and CV-optimized thresholds)
if "RF" in probas_for_curves:
    rf_proba = probas_for_curves["RF"]
    rf_t = best_t_by_model["RF"]
    rf_rep_050 = classification_report(
        y_test, (rf_proba >= 0.5).astype(int), digits=3
    )
    rf_rep_opt = classification_report(
        y_test, (rf_proba >= rf_t).astype(int), digits=3
    )
    with open(
        os.path.join(out_dir, "RF_classification_report_0.50.txt"),
        "w",
        encoding="utf-8",
    ) as f:
        f.write(rf_rep_050)
    with open(
        os.path.join(out_dir, "RF_classification_report_cvopt.txt"),
        "w",
        encoding="utf-8",
    ) as f:
        f.write(rf_rep_opt)

# LaTeX table (simple version)
try:
    mt = metrics_df.copy()
    for c in ["roc_auc", "pr_auc", "brier", "accuracy", "f1", "threshold"]:
        if c in mt.columns:
            mt[c] = mt[c].apply(
                lambda v: f"{v:.3f}" if pd.notnull(v) else ""
            )
    latex = mt.rename(
        columns={
            "label": "Setting",
            "roc_auc": "ROC AUC",
            "pr_auc": "PR AUC",
            "brier": "Brier",
            "accuracy": "Accuracy",
            "f1": "F1",
            "threshold": "Threshold",
        }
    ).to_latex(index=False, escape=False, column_format="lccccc")
    with open(
        os.path.join(tab_dir, "metrics_table_all_models.tex"),
        "w",
        encoding="utf-8",
    ) as f:
        f.write(latex)
except Exception as e:
    print("[warn] LaTeX export failed:", e)

# README-style summary of outputs
with open(os.path.join(out_dir, "README.txt"), "w", encoding="utf-8") as f:
    f.write(
        "Outputs:\n"
        "- tables/metrics_table_all_models.csv, .tex\n"
        "- tables/perm_importance_<MODEL>_percent.csv  (share%, rel_drop%, rank_percentile)\n"
        "- tables/perm_importance_all_models_percent.csv\n"
        "- figures/fig_perm_share_<MODEL>_top20.(png|pdf)\n"
        "- figures/fig_ROC_models.(png|pdf), fig_PR_models.(png|pdf)\n"
        "- figures/fig_confmat_<MODEL>_cvopt.(png|pdf)\n"
        "- figures/fig_reliability_RF.(png|pdf)\n"
    )

print("\n=== Metrics (all models) ===")
print(metrics_df)
print("\n[ok] Saved to:", out_dir)
