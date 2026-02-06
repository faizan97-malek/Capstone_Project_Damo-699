# Capstone_Project_Damo-699
Predictive maintenance using AI4I 2020 sensor data with logistic regression and tree-based models.

##  Project Overview
This capstone project explores predictive maintenance using operational and sensor-based data from the AI4I 2020 Predictive Maintenance dataset. The objective of the project is to analyze machine failure patterns and establish a foundation for developing predictive models that support proactive maintenance decision-making.

Current efforts focus on data understanding, cleaning, and exploratory data analysis (EDA) to assess data quality, identify key patterns, and evaluate dataset readiness for predictive modeling.

---
##  Current Project Structure
```text
data/
└──cleaned/
    └──ai4i2020_cleaned.csv   # Cleaned AI4I 2020 dataset
└── raw/
    └── ai4i2020.csv          # Original AI4I 2020 dataset

notebook/
└── 01_data_understanding.ipynb
└── 02_eda.ipynb
└── 03_data_cleaning.ipynb  

reports/  
└── project_proposal.pdf      # Project proposal and documentation

src/                   # Source code modules
visuals/               # Exported charts and visuals

.gitignore
LICENSE
README.md
requirements.txt
```

##  Dataset
- **Name:** AI4I 2020 Predictive Maintenance Dataset  
- **Format:** CSV  
- **Raw Location:** `data/raw/ai4i2020.csv`  
- **Cleaned Location:** `data/cleaned/ai4i2020_cleaned.csv`
- **Description:** Contains operational and sensor measurements related to machine performance and failure events.

---

##  Tools and Technologies
- Python  
- Pandas, NumPy  
- Scikit-learn  
- Matplotlib, Seaborn  
- Jupyter Notebook  
- Git & GitHub  

---

##  Setup Instructions
Install required Python dependencies using:
```bash 
pip install -r requirements.txt
```

##  Project Status
Sprint 1 is currently in progress. Data understanding, cleaning, and exploratory analysis have been completed. Hypothesis Operationalization, Validation Planning and sprint review activities are ongoing.

##  License
This project is licensed under the MIT License.

##  Note
This README will be updated as the project progresses and additional analyses, notebooks, and results are added.
