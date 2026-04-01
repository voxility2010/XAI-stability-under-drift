# XAI Robustness Under Data Drift

## Overview
This project presents a systematic evaluation of how post-hoc explainability methods behave under data drift. It focuses on the stability of SHAP and LIME explanations when the training data distribution is perturbed, while model performance remains largely unchanged.

The study demonstrates that explanation reliability can degrade significantly even when predictive accuracy is stable, highlighting a critical risk in deploying explainable machine learning systems.

---

## Problem Statement
Most machine learning systems rely on accuracy as the primary evaluation metric. However, in high-stakes domains, explanation reliability is equally important.

This project investigates:
- Whether SHAP and LIME maintain consistent feature importance rankings under drift
- How different types of data drift affect explanation stability
- Whether model accuracy is a reliable proxy for explanation quality

---

## Dataset
- Dataset: GiveMe Credit
- Number of samples: 150,000
- Task: Binary classification (credit default prediction)
- Final feature count: 19 (including engineered features)

Preprocessing:
- Missing values imputed (median and mode)
- Feature engineering applied (9 additional features)
- Z-score normalization

---

## Model
- Algorithm: XGBoost Classifier
- Training strategy: Early stopping
- Decision threshold: 0.47
- Baseline test accuracy: 93.78%

Observed accuracy range across experiments:
- Minimum: 93.32%
- Maximum: 93.81%

---

## Explainability Methods
- SHAP (KernelExplainer with KMeans background of 50 clusters)
- LIME (LimeTabularExplainer)

---

## Drift Simulation

Drift was applied only to training data to isolate model-level effects.

### Drift Types and Levels

1. Gaussian Noise
   - σ = 0.1, 0.2, 0.3

2. MCAR Missingness
   - 5%, 10%, 20% missing values

3. Covariate Shift
   - Small, Medium, Severe transformations

Total configurations evaluated: 9

---

## Evaluation Metrics

Explanation stability measured using rank correlation:

- Spearman Rank Correlation (ρ)
- Kendall Tau (τ)

Reliability threshold:
- ρ ≥ 0.7 considered stable

---

## Key Results

### 1. SHAP Stability Degradation

- Noise (σ = 0.1): ρ = 0.356
- Noise (σ = 0.3): ρ = 0.207
- Covariate shift (small): ρ = 0.133

Maximum stability loss observed:
- SHAP: 86.7%

---

### 2. LIME Stability

- Noise (σ = 0.1): ρ = 0.563
- Noise (σ = 0.3): ρ = 0.304
- Covariate shift (small): ρ = 0.502

Average stability loss:
- LIME: 38.3%
- SHAP: 49.6%

---

### 3. Missing Data Impact

MCAR missingness had minimal effect:

- SHAP: ρ ≈ 0.91
- LIME: ρ ≈ 0.80

---

### 4. Accuracy vs Explanation Reliability

- Accuracy remained within 93.3% – 93.8%
- SHAP stability dropped as low as 0.133

Conclusion:
Accuracy is not a reliable indicator of explanation quality.

---

## Extended Analysis

### KS-Test Drift Detection
- Covariate shift KS: up to 0.979
- Noise KS: ~0.26 – 0.28
- Missingness KS: ~0.04 – 0.18

Higher KS distance strongly correlates with explanation instability.

---

### SHAP–LIME Agreement
- Baseline Jaccard similarity: 1.0
- Under drift: as low as 0.667

---

### Multi-Seed Stability (5 seeds)
- SHAP (noise 0.2): ρ range 0.156 – 0.298
- LIME (noise 0.2): ρ range 0.575 – 0.681

---

### Faithfulness (Feature Deletion AUC)
- Baseline AUC: 0.8866
- Worst drift AUC: 0.8865

Observation:
Faithfulness remains stable even when ranking stability degrades.

---

## Key Insights

- Covariate shift is the most damaging drift type for explainability
- SHAP is more sensitive to distribution shifts than LIME
- Missing data has limited impact due to imputation
- Explanation stability and faithfulness are independent properties
- Accuracy alone cannot be used to validate explainability

---

## Project Structure
├── XAI_Team58_Counterfactual Chaos.ipynb
├── XAI_Robustness_Report_Team58.pdf
├── README.md

---

## How to Run

1. Clone the repository

git clone https://github.com/your-username/xai-robustness-under-data-drift.git

cd xai-robustness-under-data-drift


2. Install dependencies

pip install -r requirements.txt


3. Run the notebook

jupyter notebook


---

## Recommendations

- Deploy drift detection (e.g., KS-test) in production pipelines
- Prefer LIME in environments with frequent distribution shift
- Validate explanation stability before relying on model interpretations
- Perform multi-seed evaluation for robust conclusions

---

## Author

Independent project. Designed, implemented, and analyzed end-to-end.
