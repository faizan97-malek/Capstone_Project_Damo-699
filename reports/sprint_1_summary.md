# Sprint 1 Summary and Readiness Review

This document summarizes the outcomes of Sprint 1 and evaluates the project’s readiness to transition into the modeling and evaluation phase in Sprint 2. Sprint 1 focused on data understanding, data cleaning and validation, exploratory data analysis (EDA), and hypothesis operationalization.

---

## Review of Sprint 1 User Stories

All Sprint 1 user stories were reviewed against their acceptance criteria. The objectives of validating dataset integrity, understanding target behavior, exploring feature relationships, and operationalizing hypotheses were achieved.

---

## Key Sprint 1 Outcomes

### Dataset Validation and Cleaning

The AI4I 2020 dataset was thoroughly validated before modeling preparation.

The following checks were performed:

- Confirmed target variable (`Machine failure`) is binary (0/1)
- Verified class distribution (9,661 non-failure, 339 failure cases)
- Checked for missing values across all variables (none detected)
- Checked for duplicate records (none detected)
- Validated sensor ranges for physically implausible values (no invalid readings found)

No rows were removed, and no imputation was required.

Extreme values observed during EDA were retained because they represent realistic operational behavior rather than data errors.

A cleaned version of the dataset was generated and saved to:

`data/cleaned/ai4i2020_cleaned.csv`

The raw dataset was preserved unchanged.

---

### Exploratory Data Analysis (EDA) Insights

EDA revealed several meaningful patterns:

- The dataset is highly imbalanced, with failure events representing approximately 3.39% of observations.
- Torque and tool wear exhibit stronger separation between failure and non-failure cases.
- Rotational speed shows notable behavioral differences preceding failures.
- Air and process temperatures are strongly correlated with each other.
- Linear correlations between individual features and failure are modest, suggesting potential non-linear relationships.

These findings confirm that both linear and tree-based models are appropriate candidates for modeling.

---

### Key Decisions Made

Based on Sprint 1 findings:

- The problem is confirmed as a binary classification task.
- Logistic Regression will be used as a baseline linear model.
- Tree-based models will be used to capture non-linear relationships.
- Recall and F1-score will be prioritized over accuracy due to class imbalance.
- No aggressive outlier removal will be performed.
- Class imbalance handling will be addressed in Sprint 2.

---

## Risks and Issues Identified

Several modeling risks were identified:

- Severe class imbalance, which may bias model performance.
- Potential multicollinearity (e.g., air and process temperature).
- Risk of overfitting in tree-based models.
- The dataset is synthetic, which may limit real-world generalizability.

These risks will be addressed during feature engineering and model validation in Sprint 2.

---

## Readiness for Sprint 2

The project is ready to proceed to Sprint 2.

The dataset has been validated, cleaned, explored, and documented. Hypotheses have been operationalized, and validation metrics have been defined. Key risks have been identified and acknowledged.

Sprint 2 will focus on:

- Feature engineering
- Model development
- Model evaluation
- Class imbalance handling
- Comparative model analysis

---

## Conclusion

Sprint 1 successfully established a reliable analytical foundation. Data integrity has been confirmed, meaningful patterns have been identified, and hypotheses have been structured for empirical validation. The project is prepared to transition into predictive modeling and evaluation.
