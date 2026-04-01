# XAI Robustness Under Data Drift

Evaluating the stability of post-hoc explainability methods — SHAP and LIME — when training data distribution is perturbed while model performance remains nearly constant.

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Model](#model)
- [Explainability Methods](#explainability-methods)
- [Drift Simulation](#drift-simulation)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [Extended Analysis](#extended-analysis)
- [Key Insights](#key-insights)
- [Project Structure](#project-structure)
- [How to Run](#how-to-run)

---

## Problem Statement

This project investigates whether post-hoc explanation methods can be trusted when the data an ML model was trained on shifts over time. Specifically, it examines:

- Stability of SHAP and LIME feature importance rankings under distributional drift
- Impact of different drift types on explanation reliability
- Whether model accuracy is a valid proxy for explanation quality

---

## Dataset

**GiveMe Credit** — binary classification task (predicting credit default)

| Property | Value |
|---|---|
| Total samples | 150,000 |
| Final features | 19 |
| Task | Binary classification |

**Preprocessing:**
- Missing value imputation using median (continuous) and mode (categorical)
- Feature engineering adding 9 derived features
- Z-score normalization across all features

---

## Model

**Algorithm:** XGBoost Classifier

| Setting | Value |
|---|---|
| Training strategy | Early stopping |
| Decision threshold | 0.47 |
| Baseline accuracy | 93.78% |
| Accuracy range (all experiments) | 93.32% - 93.81% |

---

## Explainability Methods

**SHAP** — KernelExplainer with KMeans background sampling (50 clusters)

**LIME** — LimeTabularExplainer with default settings

Both methods are applied post-hoc to the trained XGBoost model to generate local feature importance explanations.

---

## Drift Simulation

Drift is applied exclusively to the training data. The test set remains unchanged throughout all experiments.

### Drift Types

| Type | Configurations |
|---|---|
| Gaussian Noise | sigma = 0.1, 0.2, 0.3 |
| MCAR Missingness | 5%, 10%, 20% |
| Covariate Shift | Small, Medium, Severe |

**Total configurations:** 9

---

## Evaluation Metrics

Explanation stability is measured by comparing feature importance rankings between baseline and drifted conditions.

| Metric | Description |
|---|---|
| Spearman Rank Correlation (rho) | Measures rank-order agreement |
| Kendall Tau | Measures pairwise rank concordance |

**Stability threshold:** rho >= 0.7

---

## Results

### SHAP Stability

| Drift Condition | Spearman rho |
|---|---|
| Gaussian Noise (sigma = 0.1) | 0.356 |
| Gaussian Noise (sigma = 0.3) | 0.207 |
| Covariate Shift (small) | 0.133 |
| MCAR Missingness | ~0.91 |

Maximum SHAP stability loss: **86.7%**
Average SHAP stability loss across all conditions: **49.6%**

### LIME Stability

| Drift Condition | Spearman rho |
|---|---|
| Gaussian Noise (sigma = 0.1) | 0.563 |
| Gaussian Noise (sigma = 0.3) | 0.304 |
| Covariate Shift (small) | 0.502 |
| MCAR Missingness | ~0.80 |

Average LIME stability loss across all conditions: **38.3%**

### Accuracy vs. Explanation Stability

| Measure | Range |
|---|---|
| Model accuracy | 93.3% - 93.8% |
| SHAP stability minimum | 0.133 |

Model accuracy remains virtually constant while explanation stability collapses — confirming that accuracy is not a reliable indicator of explanation quality.

---

## Extended Analysis

### KS-Test Drift Detection

| Drift Type | KS Statistic |
|---|---|
| Covariate shift | Up to 0.979 |
| Gaussian noise | ~0.26 - 0.28 |
| MCAR missingness | ~0.04 - 0.18 |

### SHAP-LIME Agreement (Jaccard Similarity)

| Condition | Jaccard |
|---|---|
| Baseline | 1.0 |
| Under drift | ~0.667 |

### Multi-Seed Stability (5 seeds, Gaussian Noise sigma = 0.2)

| Method | rho Range |
|---|---|
| SHAP | 0.156 - 0.298 |
| LIME | 0.575 - 0.681 |

### Faithfulness (Deletion AUC)

| Condition | Deletion AUC |
|---|---|
| Baseline | 0.8866 |
| Worst drift | 0.8865 |

Faithfulness is effectively invariant to drift conditions, indicating that stability and faithfulness measure independent properties of explanations.

---

## Key Insights

1. **Covariate shift causes maximum instability** — among all drift types, covariate shift produces the most severe degradation in explanation quality.
2. **SHAP is more sensitive than LIME** — SHAP explanations degrade faster and more severely under all drift conditions tested.
3. **Missing data has minimal effect** — due to imputation during preprocessing, MCAR missingness has a comparatively small impact on explanation stability.
4. **Stability and faithfulness are independent** — a faithful explanation is not necessarily a stable one; the two properties capture different aspects of explanation quality.
5. **Accuracy is not a proxy for explanation reliability** — model accuracy held above 93% across all experiments while SHAP stability fell as low as 0.133, demonstrating a critical blind spot in accuracy-only evaluation.

---

## Project Structure

```
.
├── XAI_Team58_Counterfactual Chaos.ipynb   # Main experiment notebook
├── XAI_Robustness_Report_Team58.pdf        # Full written report
└── README.md
```

---

## How to Run

**1. Clone the repository**

```bash
git clone https://github.com/your-username/xai-robustness-under-data-drift.git
cd xai-robustness-under-data-drift
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Launch the notebook**

```bash
jupyter notebook
```

Open `XAI_Team58_Counterfactual Chaos.ipynb` and run all cells in order.
