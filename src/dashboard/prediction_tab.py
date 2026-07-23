import pandas as pd
import plotly.express as px
import streamlit as st

from src.dashboard.prepare_input import prepare_input
from src.dashboard.predict import predict_patient


def show_prediction_tab():

    # =====================================================
    # PAGE HEADER
    # =====================================================

    st.title(
        "🩺 Clinical Readmission Risk Assessment"
    )

    st.markdown(
        """
        Estimate the probability of **30-day hospital readmission**
        using an AI-assisted clinical decision-support model.
        """
    )

    st.divider()

    # =====================================================
    # MAIN LAYOUT
    # =====================================================

    left, right = st.columns(
        [1.7, 1],
        gap="large",
    )

    # =====================================================
    # LEFT COLUMN
    # PATIENT INPUT
    # =====================================================

    with left:

        st.subheader(
            "👤 Patient Information"
        )

        # -------------------------------------------------
        # DEMOGRAPHIC INFORMATION
        # -------------------------------------------------

        st.markdown(
            "#### 🧑 Demographics"
        )

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

        st.divider()

        # -------------------------------------------------
        # HOSPITAL INFORMATION
        # -------------------------------------------------

        st.markdown(
            "#### 🏥 Hospital Information"
        )

        col_1, col_2 = st.columns(2)

        with col_1:

            time_in_hospital = st.number_input(

                "Hospital Stay (Days)",

                min_value=1,

                max_value=14,

                value=3,

            )

            n_lab_procedures = st.number_input(

                "Laboratory Procedures",

                min_value=0,

                value=40,

            )

        with col_2:

            n_procedures = st.number_input(

                "Procedures",

                min_value=0,

                value=1,

            )

            n_medications = st.number_input(

                "Number of Medications",

                min_value=1,

                value=12,

            )

        st.divider()

        # -------------------------------------------------
        # PREVIOUS HEALTHCARE VISITS
        # -------------------------------------------------

        st.markdown(
            "#### 📅 Previous Healthcare Visits"
        )

        col_1, col_2, col_3 = st.columns(3)

        with col_1:

            n_outpatient = st.number_input(

                "Outpatient",

                min_value=0,

                value=0,

            )

        with col_2:

            n_inpatient = st.number_input(

                "Inpatient",

                min_value=0,

                value=0,

            )

        with col_3:

            n_emergency = st.number_input(

                "Emergency",

                min_value=0,

                value=0,

            )

        st.divider()

        # -------------------------------------------------
        # CLINICAL INFORMATION
        # -------------------------------------------------

        st.markdown(
            "#### 🧪 Clinical Information"
        )

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

        col_1, col_2, col_3 = st.columns(3)

        with col_1:

            diag_1 = st.selectbox(

                "Primary Diagnosis",

                diagnosis_categories,

            )

        with col_2:

            diag_2 = st.selectbox(

                "Secondary Diagnosis",

                diagnosis_categories,

            )

        with col_3:

            diag_3 = st.selectbox(

                "Third Diagnosis",

                diagnosis_categories,

            )

        col_1, col_2 = st.columns(2)

        with col_1:

            glucose_test = st.selectbox(

                "Glucose Test",

                [

                    "no",

                    "normal",

                    "high",

                ],

            )

        with col_2:

            A1Ctest = st.selectbox(

                "HbA1c Test",

                [

                    "no",

                    "normal",

                    "high",

                ],

            )

        col_1, col_2 = st.columns(2)

        with col_1:

            change = st.radio(

                "Medication Changed?",

                [

                    "yes",

                    "no",

                ],

                horizontal=True,

            )

        with col_2:

            diabetes_med = st.radio(

                "Diabetes Medication?",

                [

                    "yes",

                    "no",

                ],

                horizontal=True,

            )

        st.divider()

        # -------------------------------------------------
        # PREDICTION BUTTON
        # -------------------------------------------------

        predict_button = st.button(

            "🔍 Assess Readmission Risk",

            use_container_width=True,

            type="primary",

        )

    # =====================================================
    # RIGHT COLUMN
    # RESULTS PANEL
    # =====================================================

    with right:

        st.subheader(
            "🧠 Clinical Assessment"
        )

        if predict_button:

            # =================================================
            # COLLECT PATIENT DATA
            # =================================================

            patient = {

                "age": age,

                "time_in_hospital":
                time_in_hospital,

                "n_lab_procedures":
                n_lab_procedures,

                "n_procedures":
                n_procedures,

                "n_medications":
                n_medications,

                "n_outpatient":
                n_outpatient,

                "n_inpatient":
                n_inpatient,

                "n_emergency":
                n_emergency,

                "medical_specialty":
                medical_specialty,

                "diag_1":
                diag_1,

                "diag_2":
                diag_2,

                "diag_3":
                diag_3,

                "glucose_test":
                glucose_test,

                "A1Ctest":
                A1Ctest,

                "change":
                change,

                "diabetes_med":
                diabetes_med,

                # Dummy target column
                "readmitted":
                "no",

            }

            # =================================================
            # PREPARE INPUT
            # =================================================

            patient_df = prepare_input(

                patient

            )

            # =================================================
            # PREDICTION
            # =================================================

            result = predict_patient(

                patient_df

            )

            probability = (

                result["probability"]

                * 100

            )

            risk = result["risk"]

            prediction = result["prediction"]

            shap_explanation = (

                result["shap_explanation"]

            )

            # =================================================
            # RISK CARD
            # =================================================

            if risk == "High Risk":

                st.error(

                    "🔴 HIGH READMISSION RISK"

                )

            else:

                st.success(

                    "🟢 LOW READMISSION RISK"

                )

            # =================================================
            # KEY METRICS
            # =================================================

            st.markdown(
                "### 📊 Risk Metrics"
            )

            metric_1, metric_2 = st.columns(2)

            with metric_1:

                st.metric(

                    label="Readmission Probability",

                    value=f"{probability:.1f}%",

                )

            with metric_2:

                if prediction == 1:

                    prediction_text = (
                        "High Risk"
                    )

                else:

                    prediction_text = (
                        "Low Risk"
                    )

                st.metric(

                    label="Model Classification",

                    value=prediction_text,

                )

            # =================================================
            # PROBABILITY VISUALIZATION
            # =================================================

            st.markdown(
                "### 📈 Risk Probability"
            )

            st.progress(

                min(

                    probability / 100,

                    1.0,

                )

            )

            st.caption(

                f"Decision threshold: 30% | "
                f"Estimated probability: "
                f"{probability:.1f}%"

            )

            # =================================================
            # CLINICAL RECOMMENDATION
            # =================================================

            st.markdown(
                "### 💡 Clinical Recommendation"
            )

            if risk == "High Risk":

                st.warning(

                    """
                    The model estimates an elevated probability
                    of 30-day hospital readmission.

                    Suggested actions:

                    • Consider early follow-up.

                    • Review medication adherence.

                    • Monitor the patient closely.

                    • Consider additional clinical evaluation.

                    """

                )

            else:

                st.info(

                    """
                    The model estimates a relatively lower
                    probability of 30-day hospital readmission.

                    Continue standard follow-up care
                    according to the clinical context.

                    """

                            )
            # =================================================
            # LOCAL SHAP EXPLANATION
            # =================================================

            st.divider()

            st.markdown(
                "### 🔍 Why did the model make this prediction?"
            )

            st.markdown(
                """
                SHAP explains how individual features influenced
                this specific prediction.

                • Positive values increase the estimated readmission risk.

                • Negative values reduce the estimated readmission risk.

                The chart below shows the most influential factors
                for this individual patient.
                """
            )

            # -------------------------------------------------
            # Prepare SHAP Data
            # -------------------------------------------------

            shap_df = pd.DataFrame(

                shap_explanation

            )

            # -------------------------------------------------
            # Select Top Influential Features
            # -------------------------------------------------

            shap_df["abs_shap"] = (

                shap_df["shap_value"]

                .abs()

            )

            shap_df = (

                shap_df

                .sort_values(

                    by="abs_shap",

                    ascending=False,

                )

                .head(10)

            )

            # -------------------------------------------------
            # Human-readable Feature Names
            # -------------------------------------------------

            feature_name_mapping = {

                "n_inpatient":
                "Previous Inpatient Visits",

                "n_emergency":
                "Previous Emergency Visits",

                "n_outpatient":
                "Previous Outpatient Visits",

                "time_in_hospital":
                "Hospital Stay Duration",

                "n_lab_procedures":
                "Laboratory Procedures",

                "n_procedures":
                "Number of Procedures",

                "n_medications":
                "Number of Medications",

                "age":
                "Age Group",

                "change":
                "Medication Change",

                "diabetes_med":
                "Diabetes Medication",

                "glucose_test_no":
                "Glucose Test: No",

                "glucose_test_normal":
                "Glucose Test: Normal",

                "glucose_test_high":
                "Glucose Test: High",

                "A1Ctest_no":
                "HbA1c Test: No",

                "A1Ctest_normal":
                "HbA1c Test: Normal",

                "A1Ctest_high":
                "HbA1c Test: High",

                "medical_specialty_InternalMedicine":
                "Internal Medicine",

                "medical_specialty_Emergency/Trauma":
                "Emergency / Trauma",

                "diag_1_Diabetes":
                "Primary Diagnosis: Diabetes",

                "diag_1_Circulatory":
                "Primary Diagnosis: Circulatory",

                "diag_1_Respiratory":
                "Primary Diagnosis: Respiratory",

                "diag_1_Digestive":
                "Primary Diagnosis: Digestive",

                "diag_2_Diabetes":
                "Secondary Diagnosis: Diabetes",

                "diag_3_Missing":
                "Third Diagnosis: Missing",

            }

            shap_df["display_feature"] = (

                shap_df["feature"]

                .map(feature_name_mapping)

                .fillna(

                    shap_df["feature"]

                )

            )

            # -------------------------------------------------
            # Add Contribution Direction
            # -------------------------------------------------

            shap_df["impact"] = shap_df["shap_value"].apply(

                lambda value:

                    "Increases Risk"

                    if value > 0

                    else "Reduces Risk"

            )

            # -------------------------------------------------
            # Interactive SHAP Chart
            # -------------------------------------------------

            fig = px.bar(

                shap_df.sort_values(

                    "shap_value"

                ),

                x="shap_value",

                y="display_feature",

                orientation="h",

                color="impact",

                color_discrete_map={

                    "Increases Risk":
                    "#d62728",

                    "Reduces Risk":
                    "#2ca02c",

                },

                labels={

                    "shap_value":
                    "SHAP Contribution",

                    "display_feature":
                    "Feature",

                    "impact":
                    "Impact",

                },

                title=(

                    "Top Factors Influencing "
                    "This Prediction"

                ),

            )

            fig.add_vline(

                x=0,

                line_width=1,

            )

            fig.update_layout(

                height=500,

                showlegend=True,

                yaxis_title="",

                xaxis_title="Impact on Readmission Risk",

            )

            st.plotly_chart(

                fig,

                use_container_width=True,

            )
            # =================================================
            # PATIENT SUMMARY
            # =================================================

            st.divider()

            st.markdown(
                "### 👤 Patient Summary"
            )

            with st.expander(

                "View Patient Information",

                expanded=True,

            ):

                summary_col_1, summary_col_2 = (

                    st.columns(2)

                )

                with summary_col_1:

                    st.write(

                        f"**Age:** "
                        f"{patient['age']}"

                    )

                    st.write(

                        f"**Hospital Stay:** "
                        f"{patient['time_in_hospital']} days"

                    )

                    st.write(

                        f"**Medical Specialty:** "
                        f"{patient['medical_specialty']}"

                    )

                    st.write(

                        f"**Primary Diagnosis:** "
                        f"{patient['diag_1']}"

                    )

                with summary_col_2:

                    st.write(

                        f"**Outpatient Visits:** "
                        f"{patient['n_outpatient']}"

                    )

                    st.write(

                        f"**Inpatient Visits:** "
                        f"{patient['n_inpatient']}"

                    )

                    st.write(

                        f"**Emergency Visits:** "
                        f"{patient['n_emergency']}"

                    )

                    st.write(

                        f"**Diabetes Medication:** "
                        f"{patient['diabetes_med']}"

                    )

        else:

            # =================================================
            # INITIAL STATE
            # =================================================

            st.info(

                """
                Enter patient information and click

                **Assess Readmission Risk**

                to generate a prediction.

                """

            )

            st.metric(

                label="Readmission Probability",

                value="-- %",

            )

            st.metric(

                label="Model Classification",

                value="--",

            )