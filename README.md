# 🏥 Predicting Hospital Readmission

Predicting which patients are at high risk of being readmitted to hospital within 30 days of discharge, using exploratory data analysis and machine learning.

## 📋 Overview

Hospital readmissions are a major cost driver for healthcare systems. This project analyzes patient discharge data to identify the key factors associated with readmission and builds classification models to flag high-risk patients before they leave the hospital.

**Dataset:** [Hospital Readmissions (Kaggle)](https://www.kaggle.com/datasets/dubradave/hospital-readmissions) — 10 years of patient records including demographics, hospital stay details, procedures, and medications.

## 🎯 Objectives

- Explore patterns in patient data linked to readmission risk
- Build and compare two classification models: Logistic Regression and Random Forest
- Identify the most important features driving readmission risk
- Translate findings into actionable recommendations for hospital staff

## 🛠️ Tech Stack

`Python` `Pandas` `NumPy` `Matplotlib` `Seaborn` `Scikit-learn` `Jupyter`

## 📊 Project Workflow

1. **Data Loading & Overview** — inspecting structure, types, and missing values
2. **Exploratory Data Analysis** — readmission distribution, age groups, length of stay, correlations
3. **Preprocessing** — encoding categorical variables, train/test split, feature scaling
4. **Modelling** — Logistic Regression and Random Forest classifiers
5. **Evaluation** — ROC-AUC, confusion matrices, precision/recall/F1 comparison
6. **Feature Importance** — identifying the strongest predictors of readmission

## 🔑 Key Findings

- Readmission rate in the dataset is **~47%**, making it a fairly balanced classification problem
- The strongest predictors of readmission are **number of lab procedures**, **number of medications**, and **time spent in hospital**
- Both models achieve a **ROC-AUC of ~0.63–0.65**, showing meaningful predictive signal beyond random chance
- Random Forest and Logistic Regression perform comparably, suggesting the relationship between features and readmission is largely linear

## 💡 Recommendations

- Flag patients with high medication counts or extended hospital stays for **priority follow-up care**
- Use model risk scores to **support discharge planning decisions**, not replace clinical judgement
- Future work: address class imbalance, test gradient boosting models, and incorporate diagnosis-specific features

## 📁 Repository Structure

```
├── hospital_readmission.ipynb   # Full analysis notebook
├── *.png                        # Saved EDA and model evaluation figures
└── README.md
```

## 👤 Author

**Esraa Al-Najjar**
ML & AI Researcher | Aspiring Data Scientist
[LinkedIn](https://www.linkedin.com/in/esraa-al-najjar/)
