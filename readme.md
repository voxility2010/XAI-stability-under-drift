# XAI Robustness Under Data Drift

**Evaluating Explanation Stability of SHAP and LIME Across Multiple Datasets and Model Families**

> Vardha Kathuria

---

## Overview

This notebook presents a systematic, reproducible study of how post-hoc explanation methods — **SHAP** and **LIME** — behave when the training data distribution shifts. As ML models are increasingly deployed in high-stakes domains, understanding whether their explanations remain trustworthy under real-world data drift is critical.

The core question: *If the data a model was trained on drifts, do the explanations it produces still reliably reflect the model's actual decision logic?*

Explanation stability is quantified using **Spearman rank correlation (ρ)** and **Kendall tau (τ)** between baseline and post-drift feature importances, with **ρ ≥ 0.7** as the reliability threshold (Landis & Koch, 1977).

**Key finding: SHAP is consistently more robust than LIME under all three drift types.**

---

## Datasets

| Dataset | Source | Instances | Task |
|---|---|---|---|
| **GiveMeCredit** | [Kaggle](https://www.kaggle.com/datasets/liunian394/givemesomecredit) | 150,000 | Credit default prediction |
| **Adult Income** | [UCI / OpenML](https://www.openml.org/search?type=data&id=1590) | 48,842 | Income >$50K classification |

Both datasets undergo feature engineering and Z-score normalisation before model training.

---

## Models

- **XGBoost** (gradient boosted trees) — tuned with early stopping
- **Random Forest** — 300 estimators, max depth 10

This gives **4 model × dataset combinations** evaluated across all drift configurations.

---

## Drift Simulation

Three types of drift are applied **to training data only**. The test sample is always kept clean so stability changes are attributable solely to model-level effects.

| Drift Type | Mechanism | Intensity Levels |
|---|---|---|
| **Gaussian Noise** | Additive N(0, σ) to all numerical features | σ = 0.1, 0.2, 0.3 |
| **MCAR Missingness** | Random blanking + median/mode imputation | 5%, 10%, 20% |
| **Covariate Shift** | Importance-weight resampling via logistic classifier | Small, Medium, Severe |

> **Note on Covariate Shift:** Rather than a naive additive perturbation, this implementation uses importance-weight resampling — a logistic model distinguishes training vs. shifted samples, and the resulting weights guide resampling. This is a statistically principled simulation of real-world deployment drift.

---

## Experiment Structure

```
9 drift configs × 2 datasets × 2 models = 36 experimental conditions
```

For each condition:
1. Apply drift to training data
2. Retrain the model with identical hyperparameters
3. Evaluate accuracy on the clean test set
4. Compute SHAP and LIME importances on a fixed 100-instance clean test sample
5. Calculate Spearman ρ and Kendall τ vs. baseline

---

## Extended Analysis

Beyond the main loop, four additional experiments validate the findings:

| Section | What it tests |
|---|---|
| **KS-Test Drift Detection** | Correlates KS-statistic drift magnitude with Spearman stability drop |
| **SHAP–LIME Jaccard Agreement** | Measures top-5 feature overlap between methods under drift |
| **Multi-Seed Confidence Intervals** | 5-seed Monte Carlo to confirm SHAP's advantage is statistically real |
| **Faithfulness via Feature Deletion AUC** | Progressive feature masking to check if SHAP highlights truly predictive features |

---

## Results Summary

- **SHAP maintains ρ ≥ 0.7** more consistently across all 9 drift configs, datasets, and models
- **Covariate shift** is the most destabilising drift type for both methods
- **KS distance and Spearman drop correlate strongly** (Pearson r > 0.7), confirming that distributional magnitude — not type alone — drives explanation instability
- **SHAP–LIME Jaccard < 0.5** under severe drift, meaning practitioners cannot cross-validate the two methods reliably
- **Faithfulness degrades under drift**: deletion AUC rises, indicating SHAP explanations become less faithful when training data shifts

---

## Figures Generated

| File | Description |
|---|---|
| `fig1_noise_stability.png` | Stability under Gaussian noise (all 4 model × dataset combos) |
| `fig2_all_drift_types.png` | Stability across all drift types (GiveMeCredit × XGBoost) |
| `fig3a_rank_heatmap.png` | SHAP feature rank heatmap |
| `fig3b_rank_delta.png` | SHAP feature rank delta from baseline |
| `fig4_metric_consistency.png` | Spearman ρ vs. Kendall τ scatter |
| `fig5_ks_correlation.png` | KS drift magnitude vs. stability drop |
| `fig6_jaccard.png` | SHAP–LIME cross-method agreement |
| `fig7_confidence_intervals.png` | Multi-seed confidence intervals |
| `fig8_faithfulness.png` | Feature deletion AUC curves |
| `fig9_cross_comparison.png` | Full cross-dataset × model comparison |

---

## Requirements

```bash
pip install lime xgboost scikit-learn shap
```

Other dependencies (`numpy`, `pandas`, `matplotlib`, `seaborn`, `scipy`) are standard and typically pre-installed in notebook environments.

**Tested library versions are printed at startup** for full reproducibility.

---

## Setup & Usage

### 1. Obtain the datasets

**GiveMeCredit** — download `cs-training.csv` from [Kaggle](https://www.kaggle.com/datasets/liunian394/givemesomecredit) and place it in your working directory (or update `DATA_PATH_A` in Section 2).

**Adult Income** — downloaded automatically via `sklearn.datasets.fetch_openml` (requires internet access on first run).

### 2. Run the notebook

```bash
jupyter notebook XAI-Robustness-Data-Drift.ipynb
```

Or open in [Google Colab](https://colab.research.google.com/) — update `DATA_PATH_A` to `/content/cs-training.csv` and upload the CSV when prompted.

### 3. Reproducibility

All experiments use `GLOBAL_SEED = 42`. Full results are reproducible given the same library versions (printed in Section 1.2).

---

## Notebook Structure

```
1.  Setup & Imports
2.  Dataset A — GiveMeCredit: Loading, Engineering & Model Training
3.  Dataset B — Adult Income: Loading, Engineering & Model Training
4.  Drift Simulation Engine
5.  Prediction Wrappers & Baseline XAI Values
6.  Stability Metric Helpers
7.  Main Experiment Loop (Both Datasets × Both Models)
8.  Visualizations
9.  Summary Tables
10. Extended Analysis (KS-Test · Jaccard · Confidence Intervals · Faithfulness)
11. Cross-Dataset & Cross-Model Comparison
12. Discussion, Limitations & Conclusion
```

---

## Limitations

- Only two datasets evaluated (both tabular); findings may not generalise to image, NLP, or time-series domains
- KernelSHAP uses `nsamples=200` for speed; TreeSHAP would give exact values for XGBoost
- Covariate shift is simulated via resampling — real production drift often involves concept drift (label shift) too
- Both model families are tree-based; a linear model or neural network would broaden the hypothesis class tested

---

## Citation

If you use this notebook or its methodology in your work, please cite:

> Vardha Kathuria. *XAI Robustness Under Data Drift: Evaluating Explanation Stability of SHAP and LIME Across Multiple Datasets and Model Families.* 2024.
