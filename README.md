# Capstone_Project_Damo-699  
Predictive Maintenance System using AI4I 2020 Sensor Data

## Project Overview

This capstone project focuses on building a predictive maintenance system using operational and sensor-based data from the AI4I 2020 Predictive Maintenance dataset.

The objective extends beyond simple failure prediction. The system is designed to:

- Analyze machine failure patterns
-  Build a scientifically validated machine learning pipeline
- Optimize decision thresholds for operational risk control
-  Provide explainable predictions
- Estimate session-based time-to-risk escalation (TTF proxy)
- Support proactive, risk-aware maintenance decisions

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
└── eval.json
├── gradient_boosting_pipeline.pkl
├── risk_threshold.pkl

notebook/
├── 01_data_understanding.ipynb
├── 02_eda.ipynb
├── 03_data_cleaning.ipynb
├── 04_hypothesis_validation_plan.ipynb
├── 05_failure_statistical_analysis.ipynb
├── 06_model_training_baseline.ipynb
├── SMOTE.ipynb

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
├── ttf_proxy.py

visuals/
├──Air_temperature_[K]_boxplot.png
├──Air_temperature_[K]_distribution.png
├──Process_temperature_[K]_boxplot.png
├──Process_temperature_[K]_distribution.png
├──Rotational_speed_[rpm]_boxplot.png
├──Rotational_speed_[rpm]_distribution.png
├──Tool_wear_[min]_boxplot.png
├──Tool_wear_[min]_distribution.png
├──Torque_[Nm]_boxplot.png
├──Torque_[Nm]_distribution.png
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

- Machine types
- Air temperature  
- Process temperature  
- Rotational speed  
- Torque  
- Tool wear  
- Failure indicators  

> Note: The dataset does not include actual time-to-failure timestamps. Any Remaining Useful Life (RUL) or TTF metrics referenced in later stages are proxy-based and used strictly for analytical purposes.

## Feature Engineering

The modeling pipeline incorporates engineered features to improve predictive performance and capture operational relationships, including:

- Temperature difference (Process Temperature – Air Temperature)  : Captures thermal stress
- Torque-to-RPM ratio : Captures mechanical strain per rotational speed

These features are generated consistently within the training and inference workflow to ensure consistency and reproducibility.

## Modeling Approach

The project evaluates multiple machine learning models, including:

- Logistic Regression
- Random Forest 
- Gradient Boosting

Model performance is evaluated using standard classification metrics such as precision, recall, F1-score, and ROC-AUC.

The best-performing model (Gradient Boosting) is selected based on PR-AUC and recall performance.

All preprocessing is embedded within a Scikit-learn Pipeline to prevent leakage and ensure deployment consistency.

# Threshold Optimization & Risk Tiering

Instead of using the default 0.5 classification threshold, a custom threshold is selected using Precision–Recall curve analysis to balance:

- Missed failures (false negatives)
- Unnecessary maintenance actions (false positives)

The optimized threshold enables operational risk tiers:
- High Risk → Immediate maintenance
- Medium Risk → Scheduled inspection
- Low Risk → Normal operation

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
- Time-to-risk escalation (not true survival time

## Tools and Technologies

- Python 3.11  
- Pandas & NumPy  
- Scikit-learn  
- SHAP  
- Matplotlib & Seaborn  
- VSCode  
- Git & GitHub  

## Setup

Install project dependencies using the requirements file.

Recommended Python version: **Python 3.11**

## Project Status

- Data Understanding (done) 
- Data Cleaning & EDA (done)
- Hypothesis Planning (done)
- Feature Engineering (done)
- Model Training & Evaluation (done)  
- Model Saving & Inference Layer (done) 
- SHAP Integration (in progress)  
- Streamlit Dashboard Integration (in progress)  

## License

This project is licensed under the MIT License.  
See the `LICENSE` file for more details.

## Ongoing Development

This README will continue to be updated as the project progresses. Future updates will focus on dashboard storytelling, enhanced interpretability, and final system validation. The core machine learning pipeline and threshold optimization framework are complete; remaining work centers on presentation, usability, and documentation improvements.

