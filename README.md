# Capstone_Project_Damo-699  
Predictive Maintenance System using AI4I 2020 Sensor Data

## Project Overview

This capstone project focuses on building a predictive maintenance system using operational and sensor-based data from the AI4I 2020 Predictive Maintenance dataset.

The main objective is to analyze machine failure patterns and develop a structured, reproducible machine learning pipeline capable of predicting machine failure risk. The system is designed to support proactive maintenance decision-making by identifying high-risk conditions before breakdowns occur.

The project covers the complete data science lifecycle — from data understanding and cleaning to modeling, evaluation, inference, and explainability.


## Project Structure

```text
data/
├── cleaned/
│   └── ai4i2020_cleaned.csv
├── raw/
│   └── ai4i2020.csv

models/
├── best_model.pkl
└── eval.json

notebook/
├── 01_data_understanding.ipynb
├── 02_eda.ipynb
├── 03_data_cleaning.ipynb
├── 04_hypothesis_validation_plan.ipynb
├── 05_failure_statistical_analysis.ipynb
├── 06_model_training_baseline.ipynb

reports/
└── project_proposal.pdf

src/
├── data_loader.py
├── features.py
├── labeling.py
├── preprocessing.py
├── train.py
├── inference.py
├── shap_explain.py

visuals/

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

- Machine type  
- Air temperature  
- Process temperature  
- Rotational speed  
- Torque  
- Tool wear  
- Failure indicators  

> Note: The dataset does not include actual time-to-failure timestamps. Any Remaining Useful Life (RUL) or TTF metrics referenced in later stages are proxy-based and used strictly for analytical purposes.

## Feature Engineering

The modeling pipeline incorporates engineered features to improve predictive performance and capture operational relationships, including:

- Temperature difference (Process Temperature – Air Temperature)  
- Torque-to-RPM ratio  

These features are generated consistently within the training and inference workflow to ensure reproducibility.

## Modeling Approach

The project evaluates multiple machine learning models, including:

- Logistic Regression  
- Tree-based models (e.g., Random Forest, Gradient Boosting)

Model performance is evaluated using standard classification metrics such as precision, recall, F1-score, and ROC-AUC.

The trained model pipeline is saved to ensure consistent preprocessing and prediction during inference.

## Model Explainability

SHAP (SHapley Additive Explanations) is integrated to interpret model predictions and identify the most influential features driving failure risk.

This allows:

- Understanding which features contribute most to individual predictions  
- Interpreting model behavior beyond raw accuracy metrics  
- Providing transparency in predictive maintenance decisions  

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
- Streamlit Dashboard Integration (planned)  

## License

This project is licensed under the MIT License.  
See the `LICENSE` file for more details.

## Ongoing Development

This README will continue to be updated as the project progresses. Future updates will include expanded model comparisons, additional evaluation insights, dashboard integration, and enhanced explainability components.
