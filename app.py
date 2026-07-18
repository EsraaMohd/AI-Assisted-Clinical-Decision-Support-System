from pathlib import Path

import streamlit as st

from src.dashboard.prepare_input import prepare_input

from src.dashboard.predict import predict_patient
# -------------------------------------------------------
# Page Configuration
# -------------------------------------------------------

st.set_page_config(
    page_title="AI Clinical Decision Support",
    page_icon="🏥",
    layout="wide",
)

# -------------------------------------------------------
# Header
# -------------------------------------------------------

st.title("🏥 AI-Assisted Clinical Decision Support System")

st.markdown(
"""
Predict the probability of **30-day hospital readmission**
using Machine Learning.

This system was developed as an AI project for
clinical decision support.
"""
)

st.divider()

# -------------------------------------------------------
# Sidebar
# -------------------------------------------------------

with st.sidebar:

    st.header("📊 Project Information")

    st.success("Best Model")

    st.write("**Tuned XGBoost**")

    st.metric(
        label="Recall",
        value="84.6%",
    )

    st.metric(
        label="ROC-AUC",
        value="0.637",
    )

    st.metric(
        label="Threshold",
        value="0.30",
    )

    st.divider()

    st.write(
        """
        **Workflow**

        ✅ Data Cleaning

        ✅ Feature Engineering

        ✅ Model Training

        ✅ Hyperparameter Tuning

        ✅ Threshold Optimization

        ✅ Probability Calibration
        """
    )

# -------------------------------------------------------
# Main Layout
# -------------------------------------------------------

left, right = st.columns([2, 1])

# -------------------------------------------------------
# Left Column
# -------------------------------------------------------

with left:

   # st.subheader("🩺 Patient Information")

    st.subheader("👤 Patient Information")

    age = st.selectbox(
         "Age",
        [
            "[40-50)",
            "[50-60)",
            "[60-70)",
            "[70-80)",
            "[80-90)",
         "[90-100)",
        ],
    )

    medical_specialty = st.selectbox(
        "Medical Specialty",
        [
            "Missing",
            "InternalMedicine",
            "Emergency/Trauma",
            "Family/GeneralPractice",
            "Cardiology",
            "Surgery",
             "Other",
        ],
    )
    st.subheader("🏥 Hospital Information")

    time_in_hospital = st.number_input(
        "Hospital Stay (Days)",
        min_value=1,
        max_value=14,
        value=3,
    )

    n_lab_procedures = st.number_input(
        "Lab Procedures",
        min_value=0,
        value=40,
    )

    n_procedures = st.number_input(
        "Procedures",
        min_value=0,
        value=1,
    )

    n_medications = st.number_input(
        "Medications",
        min_value=1,
        value=12,
    )

    st.subheader("📅 Previous Visits")

    n_outpatient = st.number_input(
        "Outpatient Visits",
        min_value=0,
        value=0,
    )

    n_inpatient = st.number_input(
        "Previous Inpatient Visits",
        min_value=0,
        value=0,
    )

    n_emergency = st.number_input(
        "Emergency Visits",
        min_value=0,
        value=0,
    )

    st.subheader("🧪 Clinical Information")

    diagnosis_categories = [
        "Circulatory",
        "Respiratory",
        "Digestive",
        "Diabetes",
        "Injury",
        "Musculoskeletal",
        "Other",
        "Missing",
    ]

    diag_1 = st.selectbox(
        "Primary Diagnosis",
        diagnosis_categories,
    )

    diag_2 = st.selectbox(
        "Secondary Diagnosis",
        diagnosis_categories,
    )

    diag_3 = st.selectbox(
        "Third Diagnosis",
        diagnosis_categories,
    )

    glucose_test = st.selectbox(
        "Glucose Test",
        [
            "no",
            "normal",
            "high",
        ],
    )

    A1Ctest = st.selectbox(
        "HbA1c Test",
        [
            "no",
            "normal",
            "high",
        ],
    )

    change = st.radio(
        "Medication Changed?",
        [
            "yes",
            "no",
        ],
    )

    diabetes_med = st.radio(
        "Diabetes Medication?",
        [
            "yes",
            "no",
        ],
    )

    predict_button = st.button(
    "🔍 Predict Readmission Risk",
    use_container_width=True,
   )
    
# -------------------------------------------------------
# Right Column
# -------------------------------------------------------

with right:

    st.subheader("🤖 Prediction")

    if predict_button:

        # ---------------------------------
        # Collect Patient Data
        # ---------------------------------

        patient = {

            "age": age,

            "time_in_hospital": time_in_hospital,

            "n_lab_procedures": n_lab_procedures,

            "n_procedures": n_procedures,

            "n_medications": n_medications,

            "n_outpatient": n_outpatient,

            "n_inpatient": n_inpatient,

            "n_emergency": n_emergency,

            "medical_specialty": medical_specialty,

            "diag_1": diag_1,

            "diag_2": diag_2,

            "diag_3": diag_3,

            "glucose_test": glucose_test,

            "A1Ctest": A1Ctest,

            "change": change,

            "diabetes_med": diabetes_med,

            # Dummy target
            "readmitted": "no",

        }

        # ---------------------------------
        # Prepare Features
        # ---------------------------------

        patient_df = prepare_input(
            patient
        )

        # ---------------------------------
        # Prediction
        # ---------------------------------

        result = predict_patient(
            patient_df
        )

        # ---------------------------------
        # Display Results
        # ---------------------------------

        probability = result["probability"] * 100

        if result["risk"] == "High Risk":

            st.error("🔴 High Risk")

        else:

            st.success("🟢 Low Risk")

        st.metric(

            label="Readmission Probability",

            value=f"{probability:.1f}%",

        )

        st.metric(

            label="Prediction",

            value=result["prediction"],

        )

    else:

        st.warning(
            "No prediction yet."
        )

        st.metric(

            label="Readmission Probability",

            value="-- %",

        )

    st.subheader("🤖 Prediction")

    st.warning(
        "No prediction yet."
    )

    st.metric(
        label="Probability",
        value="-- %",
    )

st.divider()

st.caption(
    "Developed using Python • Scikit-learn • XGBoost • Streamlit"
)