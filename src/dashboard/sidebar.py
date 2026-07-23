import streamlit as st


def show_sidebar():

    with st.sidebar:

        st.header("📊 Project Information")

        st.success("Best Model")

        st.write("**Tuned XGBoost**")

        st.metric(

            "Recall",

            "84.6%"

        )

        st.metric(

            "ROC-AUC",

            "0.637"

        )

        st.metric(

            "Threshold",

            "0.30"

        )

        st.divider()

        st.subheader("Pipeline")

        st.write("✅ Data Cleaning")

        st.write("✅ Feature Engineering")

        st.write("✅ Model Training")

        st.write("✅ Hyperparameter Tuning")

        st.write("✅ Threshold Optimization")

        st.write("✅ Calibration")

        st.write("✅ Dashboard")

        st.divider()

        st.caption(

            "AI Clinical Decision Support"

        )