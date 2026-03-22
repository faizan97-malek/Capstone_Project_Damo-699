# Capstone_Project_Damo-699  
Predictive Maintenance System using AI4I 2020 Sensor Data

## Project Overview

This capstone project focuses on building a predictive maintenance system using operational and sensor-based data from the AI4I 2020 Predictive Maintenance dataset.

The system goes beyond binary failure prediction. It:

- Detects at-risk machines before failure using a calibrated ensemble model (PR-AUC 0.873, Recall 95.6%)
- Explains every prediction via SHAP feature importance
- Estimates time-to-failure using a regression model and a trend-based proxy
- Runs a live simulation dashboard degrading machines tick by tick across the full fleet
- Conducts survival analysis (Kaplan-Meier + Cox PH) to model failure timelines
- Persists simulation state across sessions via a snapshot system

Rather than only predicting failure (0/1), the system converts model probabilities into:
- Risk tiers (Low / Medium / High)
- Optimized maintenance thresholds
- Feature-level explanations
- Trend-based time-to-risk estimates


## Project Structure

```text
app/
├── streamlit_app.py
data/
├── cleaned/
│   └── ai4i2020_cleaned.csv
├── raw/
│   └── ai4i2020.csv

models/
├── best_model.pkl
├── eval.json
├── gradient_boosting_pipeline.pkl
├── nb06_baseline_gb.pkl
├── risk_threshold.pkl
├── rul_regressor.joblib
├── threshold.json

notebook/
├── 01_data_understanding.ipynb
├── 02_eda.ipynb
├── 03_data_cleaning.ipynb
├── 03b_smote_experiment.ipynb
├── 04_hypothesis_testing.ipynb
├── 05_failure_statistical_analysis.ipynb
├── 06_model_training_baseline.ipynb
├── 07_ensemble_deep_learning_comparison.ipynb
├── 08_model_explainability_SHAP.ipynb
├──09_rul_ttf_regression.ipynb
├──10_survival_analysis1.ipynb

reports/
└── project_proposal.pdf
└── Project Proposal_Adjusted_Hypothesis.docx

src/
├── __init__.py
├── dashboard_utils.py
├── data_loader.py
├── features.py
├── inference.py
├── labeling.py
├── preprocessing.py
├── shap_explain.py
├── simulator.py
├── survival_analysis.py
├── train.py
├── ttf_trend.py

visuals/
├──Air_temperature_[K]_boxplot.png
├──Air_temperature_[K]_distribution.png
├──air_temperature_k_vs_machine_failure.png
├──correlation_heatmap.png
├──failure_type_distribution.png
├──machine_failure_distribution.png
├──machine_type_vs_machine_failure.png
├──missing_values_heatmap.png
├──Process_temperature_[K]_boxplot.png
├──Process_temperature_[K]_distribution.png
├──process_temperature_k_vs_machine_failure.png
├──Rotational_speed_[rpm]_boxplot.png
├──Rotational_speed_[rpm]_distribution.png
├──rotational_speed_rpm_vs_machine_failure.png
├──sensor_feature_distribution.png
├──shap_bar.png
├──shap_by_tier.png
├──shap_dependencies.png
├──shap_importance_bar.png
├──shap_local_fn.png
├──shap_local_tn.png
├──shap_local_tp.png
├──shap_summary.png
├──Tool_wear_[min]_boxplot.png
├──Tool_wear_[min]_distribution.png
├──tool_wear_min_vs_machine_failure.png
├──top_feature_importances_clean.png
├──Torque_[Nm]_boxplot.png
├──Torque_[Nm]_distribution.png
├──torque_nm_vs_machine_failure.png

.gitignore
LICENSE
README.md
requirements.txt
```
---

## Dataset

- **Name:** AI4I 2020 Predictive Maintenance Dataset  
- **Format:** CSV  
- **Raw Location:** `data/raw/ai4i2020.csv`  
- **Cleaned Location:** `data/cleaned/ai4i2020_cleaned.csv`  

The dataset contains operational and sensor measurements related to machine performance and failure events. Key variables include:

- Machine types - H (high), M (medium), L (low)
- Air temperature [K]
- Process temperature [K]  
- Rotational speed [rpm]
- Torque [Nm]
- Tool wear [min] 
- Failure indicators  

Leakage columns (TWF, HDF, PWF, OSF, RNF) are removed before modeling.

> Note: The dataset does not include actual time-to-failure timestamps. Any Remaining Useful Life (RUL) or TTF metrics referenced in later stages are proxy-based and used strictly for analytical purposes.

## Feature Engineering

The modeling pipeline incorporates engineered features to improve predictive performance and capture operational relationships, including:

- Temperature difference (Process Temperature – Air Temperature)  : Captures thermal stress
- Torque-to-RPM ratio : Captures mechanical strain per rotational speed

These features are generated consistently within the training and inference workflow to ensure consistency and reproducibility.

## Modeling Approach

12 models were evaluated in **Notebook 07** using a stratified **80/20 train-test split** (`random_state=42`).

### Model Performance

| Model | PR-AUC | ROC-AUC |
|------|--------|--------|
| **Weighted Soft Voting (RF + GB + MLP)** | **0.873** | **0.973** |
| Stacking (RF + GB + MLP → LR meta) | 0.872 | 0.973 |
| Gradient Boosting | 0.870 | 0.972 |
| XGBoost | 0.859 | 0.972 |
| Random Forest | 0.826 | 0.955 |
| MLP | 0.820 | 0.980 |
| TabNet | 0.697 | 0.962 |
| LSTM | 0.600 | 0.963 |
| Logistic Regression | 0.446 | 0.934 |

---

### Deployed Model
- **Calibrated Weighted Soft Voting Ensemble**
- Models: Random Forest + Gradient Boosting + MLP  
- **Weights:** `[1, 2, 2]`  
- **Calibration:** Isotonic  

**Why Soft Voting over Stacking?**
- Produces **well-distributed probabilities (0.0–0.96)**
- Enables meaningful **continuous risk visualization**
- Stacking compresses safe predictions (~0.12–0.13), making dashboards less interpretable  

---

### Threshold Tuning
- Optimized on **Precision-Recall curve**
- Target: **Recall ≥ 95%**
- Final threshold: **≈ 0.022**

---

## Statistical Hypothesis Testing

All hypotheses were tested and confirmed.  
Full implementation: `notebook/04_hypothesis_testing.ipynb`

### Results Summary

| Hypothesis | Description | Test | Result |
|----------|-------------|------|--------|
| **H1** | Temp_diff × Torque interaction drives failure | Logistic Regression (interaction term) | ✅ Confirmed |
| **H2** | Sensor variables impact failure | One-Way ANOVA | ✅ Confirmed |
| **H3** | Ensembles outperform other models | Friedman Test (χ²=26.55, p < 0.001) | ✅ Confirmed |

---


## Setup
### 1. Clone the Repository

```bash
git clone https://github.com/faizan97-malek/Capstone_Project_Damo-699.git
```

### 2. Create & Activate Virtual Environment (optional but recommended)
```bash
python -m venv venv
```

#### Windows
```bash
venv\Scripts\activate
```

#### Mac/Linux
```bash
source venv/bin/activate
```

### 3. Install project dependencies
```bash
pip install -r requirements.txt
```  

### 4. Train all models
```bash
python -m src.train
```

### 5. Run Dashboard

From the project root:

```bash
streamlit run app/streamlit_app.py
```
This opens up  http://localhost:8501 where u can see the dashboard.


## Dashboard

The system includes a **3-page interactive Streamlit dashboard** for real-time monitoring and analysis.

---

### 🔹 Page 1 — Simulation Dashboard
- Simulates **real-time machine degradation** across the fleet (10,000 machines)
- Displays:
  - Risk probability gauge  
  - SHAP top-5 feature drivers  
  - Sensor readings table  
  - Time-to-failure estimate  
  - High-risk alerts  
  -  Machines under maintenance  

 **Automatic Maintenance Logic**
- Triggered at **≥ 90% risk**
- Machine enters maintenance state  
- Resets after cooldown:
  - Tool wear → 0  
  - Sensors → baseline  

---

### 🔹 Page 2 — What-If Analysis
- Modify any machine’s sensor values  
- Instantly updates:
  - Risk prediction  
  - Failure probability  
- Useful for **scenario testing and decision-making**

---

### 🔹 Page 3 — Survival Analysis
- Kaplan-Meier survival curves  
- Filters:
  - Machine type  
  - Tool wear bins  
  - Type + wear bins
- Cox Proportional Hazards:
  - Individual survival curves  
  - Hazard ratios with significance  
  - Cox Coefficients (Log Hazard)

---

### Snapshot System
- On **Pause** → saves state to: data/latest_snapshot.csv
- Stores:
    - All 10,000 machines  
    - Updated simulated values  
    - Unchanged machines preserved  

- On **Start**:
    - Automatically resumes from last data saved to latest_snapshot.csv


## Threshold Optimization & Risk Tiering

Instead of using the default 0.5 classification threshold, a custom threshold is selected using Precision–Recall curve analysis to balance:

- Missed failures (false negatives)
- Unnecessary maintenance actions (false positives)

The threshold is saved as a deployment artifact (risk_threshold.pkl) and used during inference.

## Model Explainability

SHAP (SHapley Additive Explanations) is integrated to interpret model predictions and identify the most influential features driving failure risk.

This allows:

- Understanding which features contribute most to individual predictions  
- Interpreting model behavior beyond raw accuracy metrics  
- Providing transparency in predictive maintenance decisions  

## Time-to-Risk Proxy (TTF Proxy)

Because the dataset lacks true survival durations, the project implements a session-based Time-to-Failure proxy using:

- Risk probability trend modeling (linear slope)
- Threshold-crossing estimation
- Tool wear adjustment
- Horizon capping safeguards

This provides an interpretable estimate of:
- Time-to-risk escalation (not true survival time)

## Tools and Technologies

- Python
- Streamlit  
- VSCode  
- Deep Learning and Machine Learning
- Survival Analysis
- Git & GitHub 
- Taiga 

## License

This project is licensed under the MIT License.  
See the `LICENSE` file for more details.
