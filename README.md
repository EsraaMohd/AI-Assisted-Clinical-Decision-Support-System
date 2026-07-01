# 🏥 AI-Assisted Clinical Decision Support System for Hospital Readmission Risk Assessment

> *Predicting which patients are at high risk of being readmitted within 30 days of discharge — and explaining why.*

![Python](https://img.shields.io/badge/Python-3.9-blue) ![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0-orange) ![Pandas](https://img.shields.io/badge/Pandas-1.3-green) ![Status](https://img.shields.io/badge/Status-Active%20Development-brightgreen)

---

## 📌 Project Overview

Hospital readmission within 30 days of discharge is one of the most measurable and costly inefficiencies in modern healthcare. Research consistently shows that a significant proportion of readmissions are preventable — if the right patients are identified early enough to receive targeted follow-up care.

This project presents the **first version of an AI-assisted clinical decision support prototype** that combines predictive modelling with explainability to help healthcare professionals identify high-risk patients at the point of discharge.

The goal is not simply to build a model that predicts readmission — but to design a system that clinicians can **understand, trust, and act upon**.

> *"A readmission prediction model that clinicians do not trust or understand is a failed solution — regardless of its AUC score."*

---

## 🎯 Problem Statement

| Fact | Impact |
|---|---|
| ~20% of Medicare patients are readmitted within 30 days | High financial cost to healthcare systems |
| Many readmissions are preventable | Targeted follow-up can reduce risk significantly |
| Clinicians lack real-time risk signals at discharge | Decisions are made without data-driven support |

---

## 🏗️ System Architecture

```
Electronic Health Records (Patient Data)
              │
              ▼
    ┌─────────────────────┐
    │  Data Preprocessing  │
    │  & Feature Engineering│
    └─────────────────────┘
              │
              ▼
    ┌─────────────────────┐
    │   ML Prediction      │
    │   Pipeline           │
    │  (LR + RF + GB)      │
    └─────────────────────┘
              │
              ▼
    ┌─────────────────────┐
    │  Explainability      │  ← [In Development: SHAP]
    │  Layer               │
    └─────────────────────┘
              │
              ▼
    ┌─────────────────────┐
    │  Clinical Dashboard  │  ← [In Development: Streamlit]
    │  & REST API          │  ← [In Development: FastAPI]
    └─────────────────────┘
              │
              ▼
    Healthcare Professional
    (Discharge Planning Decision)
```

---

## 📊 Dataset

| Property | Value |
|---|---|
| **Source** | [Kaggle — Predicting Hospital Readmissions](https://www.kaggle.com/datasets/dubradave/hospital-readmissions) |
| **Records** | ~25,000 patient discharge records |
| **Features** | Age, time in hospital, procedures, medications, diagnoses, lab results |
| **Target** | Binary — readmitted within 30 days (Yes/No) |
| **Readmission Rate** | ~47% (relatively balanced dataset) |

---

## 🔬 Methodology

### Stage 1 — Exploratory Data Analysis
- Distribution analysis of readmission rates across age groups, diagnoses, and hospital stay duration
- Correlation analysis between clinical variables and readmission risk
- Identification of key risk factors through statistical exploration

### Stage 2 — Preprocessing & Feature Engineering
- Categorical encoding of diagnosis codes and demographic variables
- Train/test split with stratification to preserve class balance
- Feature scaling for linear models

### Stage 3 — Predictive Modelling
Two classification models trained and compared:

| Model | Strengths |
|---|---|
| Logistic Regression | Interpretable, fast, strong baseline |
| Random Forest | Captures non-linear patterns, feature importance |

### Stage 4 — Evaluation
Models evaluated across multiple metrics to reflect clinical priorities:
- **ROC-AUC** — overall discrimination ability
- **Recall** — ability to catch high-risk patients (critical in clinical settings)
- **Precision** — avoiding unnecessary alerts
- **F1 Score** — balance between precision and recall

---

## 📈 Results

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.60 | 0.63 | 0.40 | 0.49 | 0.65 |
| Random Forest | 0.60 | 0.59 | 0.51 | 0.55 | 0.63 |

**Key insight:** Random Forest achieved higher Recall — meaning it identifies more at-risk patients, which is the clinically preferred outcome even at the cost of some precision.

### Top Predictive Features (Random Forest)

The most important features driving readmission risk:
1. **n_lab_procedures** — number of lab tests performed
2. **n_medications** — number of medications administered
3. **time_in_hospital** — length of hospital stay
4. **age** — patient age bracket
5. **medical_specialty** — referring specialty

---

## 🔭 Future Industrial Extensions

This prototype represents **Phase 1** of a planned multi-phase development roadmap toward a production-ready clinical decision support system:

### Phase 2 — Explainable AI (In Development)
- **SHAP integration** to explain individual predictions to clinicians
- Feature-level explanations: *"This patient's risk is elevated primarily due to high medication count and previous admissions"*
- Fairness analysis across demographic groups

### Phase 3 — Clinical Dashboard (Planned)
- **Streamlit** interactive interface for discharge planning
- Patient-level risk scores with visual explanation
- Clinician-friendly design tested with healthcare workflow requirements

### Phase 4 — API & Deployment (Planned)
- **FastAPI** REST endpoint for integration with Hospital Information Systems
- **Docker** containerisation for reproducible deployment
- FHIR-compatible data interface for real-world EHR integration

### Phase 5 — Responsible AI & Governance (Planned)
- Bias detection across age, gender, and diagnosis groups
- Privacy-preserving design aligned with GDPR requirements
- Model transparency documentation for clinical stakeholders
- Iterative validation with clinical staff feedback

---

## ⚠️ Responsible AI Considerations

This project acknowledges the following considerations for responsible deployment in clinical settings:

- **Explainability:** Model predictions should always be accompanied by feature-level explanations to support (not replace) clinical judgement
- **Fairness:** Readmission risk models can reflect historical biases in care delivery; bias auditing is a planned next step
- **Privacy:** Patient data must be handled in compliance with GDPR and healthcare data regulations
- **Human oversight:** This system is designed as a decision *support* tool — final discharge decisions remain with the clinician

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Data Analysis | Python, Pandas, NumPy |
| Visualisation | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn |
| Notebook Environment | Jupyter |
| Version Control | Git / GitHub |
| Planned | SHAP, Streamlit, FastAPI, Docker |

---

## 📁 Repository Structure

```
hospital-readmission-prediction/
│
├── hospital_readmission.ipynb    # Full analysis notebook (EDA + Modelling)
├── *.png                         # EDA and model evaluation figures
└── README.md
```

*Repository structure will expand in Phase 2–5 to include `/app`, `/dashboard`, `/api`, `/explainability`, and `/docker`.*

---

## 👤 Author

**Esraa Al-Najjar**
ML & AI Researcher | Aspiring Data Scientist
[LinkedIn](https://linkedin.com/in/esraa-al-najjar-697b39149) | [GitHub](https://github.com/EsraaMohd)

---

## 📚 References

- Rajkomar, A. et al. (2018). Scalable and accurate deep learning with electronic health records. *NPJ Digital Medicine.*
- Frizzell, J.D. et al. (2017). Prediction of 30-day all-cause readmissions in patients hospitalized for heart failure. *JAMA Cardiology.*
- Arrieta, A.B. et al. (2020). Explainable Artificial Intelligence (XAI): Concepts, taxonomies, opportunities and challenges. *Information Fusion.*
