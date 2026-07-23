import streamlit as st


def show_about_tab():

    st.title(
        "ℹ️ About the System"
    )

    st.markdown(
        """
        ## AI-Assisted Clinical Decision Support System

        This system estimates the probability of **30-day hospital
        readmission** using machine learning.

        The system is designed as a **clinical decision-support tool**,
        not as a replacement for professional medical judgment.
        """
    )

    st.divider()

    # =====================================================
    # SYSTEM WORKFLOW
    # =====================================================

    st.subheader(
        "🔄 Machine Learning Workflow"
    )

    workflow = [

        "Data Loading",

        "Data Preprocessing",

        "Feature Engineering",

        "Model Training",

        "Hyperparameter Tuning",

        "Threshold Optimization",

        "Probability Calibration",

        "Risk Prediction",

    ]

    for index, step in enumerate(

        workflow,

        start=1,

    ):

        st.write(

            f"**{index}.** {step}"

        )

    st.divider()

    # =====================================================
    # TECHNOLOGIES
    # =====================================================

    st.subheader(
        "🛠️ Technologies"
    )

    col_1, col_2, col_3 = st.columns(3)

    with col_1:

        st.markdown(
            """
            **Programming**

            🐍 Python

            🐼 Pandas

            🔢 NumPy
            """
        )

    with col_2:

        st.markdown(
            """
            **Machine Learning**

            🤖 Scikit-learn

            ⚡ XGBoost

            📊 Matplotlib
            """
        )

    with col_3:

        st.markdown(
            """
            **Application**

            🌐 Streamlit

            🐳 Docker

            📦 Joblib
            """
        )

    st.divider()

    # =====================================================
    # IMPORTANT DISCLAIMER
    # =====================================================

    st.subheader(
        "⚠️ Important Disclaimer"
    )

    st.warning(
        """
        This application provides machine learning-based risk estimation
        for research and decision-support purposes.

        It is not intended to replace qualified medical professionals,
        clinical assessment, or physician judgment.
        """
    )