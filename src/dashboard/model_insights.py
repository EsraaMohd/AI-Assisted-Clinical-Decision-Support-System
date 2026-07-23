import streamlit as st

import pandas as pd

import plotly.express as px

import plotly.graph_objects as go

from pathlib import Path


def show_model_insights():

    st.title(
        "📊 Model Insights"
    )

    st.markdown(
        """
        Explore the performance, decision threshold, confusion matrix,
        calibration behavior, and comparative performance of the
        machine learning models.
        """
    )

    st.divider()

    # =====================================================
    # MODEL OVERVIEW
    # =====================================================

    st.subheader(
        "🤖 Selected Model"
    )

    st.info(
        """
        The final deployed model is a tuned **XGBoost Classifier**.

        The model was selected after comparing multiple machine learning
        algorithms and optimizing the decision threshold for improved
        detection of potential hospital readmissions.
        """
    )

    st.divider()

    # =====================================================
    # PERFORMANCE METRICS
    # =====================================================

    st.subheader(
        "📈 Tuned XGBoost Performance"
    )

    col_1, col_2, col_3, col_4 = st.columns(4)

    with col_1:

        st.metric(
            "Accuracy",
            "59.8%",
        )

    with col_2:

        st.metric(
            "Precision",
            "58.0%",
        )

    with col_3:

        st.metric(
            "Recall",
            "52.4%",
        )

    with col_4:

        st.metric(
            "ROC-AUC",
            "0.632",
        )

    st.caption(
        """
        These metrics correspond to the tuned XGBoost model evaluated
        using the default threshold of 0.50.
        """
    )

    st.divider()

    # =====================================================
    # MODEL COMPARISON
    # =====================================================

    st.subheader(
        "⚖️ Model Comparison"
    )

    comparison_path = Path(

        "reports/model_comparison.csv"

    )

    if comparison_path.exists():

        comparison_df = pd.read_csv(

            comparison_path

        )

        st.dataframe(

            comparison_df,

            use_container_width=True,

            hide_index=True,

        )

        # ---------------------------------------------
        # SELECT METRIC
        # ---------------------------------------------

        selected_metric = st.selectbox(

            "Select performance metric",

            [

                "Accuracy",

                "Precision",

                "Recall",

                "F1",

                "ROC-AUC",

            ],

            key="model_comparison_metric",

        )

        # ---------------------------------------------
        # INTERACTIVE BAR CHART
        # ---------------------------------------------

        chart_df = comparison_df.dropna(

            subset=[selected_metric]

        )

        fig = px.bar(

            chart_df,

            x="Model",

            y=selected_metric,

            text_auto=".3f",

            title=(

                f"{selected_metric} Comparison"

            ),

        )

        fig.update_layout(

            yaxis_range=[

                0,

                1,

            ],

            xaxis_title="Model",

            yaxis_title=selected_metric,

        )

        st.plotly_chart(

            fig,

            use_container_width=True,

        )

        # ---------------------------------------------
        # BEST MODEL BY SELECTED METRIC
        # ---------------------------------------------

        best_model_row = chart_df.loc[

            chart_df[selected_metric].idxmax()

        ]

        st.success(

            f"""
            Best model based on **{selected_metric}**:

            **{best_model_row['Model']}**

            Score: **{best_model_row[selected_metric]:.3f}**
            """

        )

    else:

        st.warning(

            "Model comparison file not found."

        )

    st.divider()

    # =====================================================
    # THRESHOLD ANALYSIS
    # =====================================================

    st.subheader(
        "🎯 Decision Threshold Analysis"
    )

    st.markdown(
        """
        The decision threshold controls when a predicted probability
        is converted into a positive readmission-risk classification.

        Because this project prioritizes identifying potential
        readmission cases, a lower threshold can improve sensitivity.
        """
    )

    threshold_path = Path(

        "reports/threshold_results.csv"

    )

    if threshold_path.exists():

        threshold_df = pd.read_csv(

            threshold_path

        )

        st.dataframe(

            threshold_df,

            use_container_width=True,

            hide_index=True,

        )

        selected_metric = st.selectbox(

            "Select threshold metric",

            [

                "Recall",

                "Precision",

                "F1",

                "Accuracy",

            ],

            key="threshold_metric",

        )

        fig = px.line(

            threshold_df,

            x="Threshold",

            y=selected_metric,

            markers=True,

            title=(

                f"{selected_metric} "
                "Across Classification Thresholds"

            ),

        )

        fig.update_layout(

            xaxis_title="Threshold",

            yaxis_title=selected_metric,

            yaxis_range=[

                0,

                1,

            ],

            hovermode="x unified",

        )

        fig.add_vline(

            x=0.30,

            line_dash="dash",

            annotation_text="Selected Threshold: 0.30",

        )

        st.plotly_chart(

            fig,

            use_container_width=True,

        )

        selected_row = threshold_df.loc[

            threshold_df["Threshold"] == 0.30

        ]

        if not selected_row.empty:

            selected_row = selected_row.iloc[0]

            st.success(

                f"""
                The selected threshold is **0.30**.

                At this threshold:

                • Recall: **{selected_row['Recall']:.3f}**

                • Precision: **{selected_row['Precision']:.3f}**

                • F1-score: **{selected_row['F1']:.3f}**

                • Accuracy: **{selected_row['Accuracy']:.3f}**
                """

            )

    else:

        st.warning(

            "Threshold analysis file not found."

        )

    st.divider()

    # =====================================================
    # CONFUSION MATRIX
    # =====================================================

    st.subheader(

        "🔲 Confusion Matrix"

    )

    st.markdown(

        """
        The confusion matrix shows how the model's predictions compare
        with the actual readmission outcomes.
        """

    )

    confusion_path = Path(

        "reports/confusion_matrix_threshold_030.csv"

    )

    if confusion_path.exists():

        confusion_df = pd.read_csv(

            confusion_path

        )

        matrix = confusion_df.pivot(

            index="Actual",

            columns="Predicted",

            values="Count",

        )

        matrix = matrix.reindex(

            index=[

                "No Readmission",

                "Readmission",

            ],

            columns=[

                "No Readmission",

                "Readmission",

            ],

        )

        fig = px.imshow(

            matrix,

            text_auto=True,

            aspect="auto",

            title=(

                "Confusion Matrix "
                "(Threshold = 0.30)"

            ),

            labels={

                "x": "Predicted",

                "y": "Actual",

                "color": "Patients",

            },

        )

        st.plotly_chart(

            fig,

            use_container_width=True,

        )

        tn = matrix.loc[

            "No Readmission",

            "No Readmission",

        ]

        fp = matrix.loc[

            "No Readmission",

            "Readmission",

        ]

        fn = matrix.loc[

            "Readmission",

            "No Readmission",

        ]

        tp = matrix.loc[

            "Readmission",

            "Readmission",

        ]

        col_1, col_2, col_3, col_4 = st.columns(4)

        with col_1:

            st.metric(

                "True Negatives",

                int(tn),

            )

        with col_2:

            st.metric(

                "False Positives",

                int(fp),

            )

        with col_3:

            st.metric(

                "False Negatives",

                int(fn),

            )

        with col_4:

            st.metric(

                "True Positives",

                int(tp),

            )

    else:

        st.warning(

            "Confusion matrix file not found."

        )

    st.divider()

    # =====================================================
    # CALIBRATION CURVE
    # =====================================================

    st.subheader(

        "📉 Probability Calibration"

    )

    st.markdown(

        """
        The calibration curve evaluates how closely predicted
        probabilities correspond to observed readmission frequencies.
        """

    )

    calibration_plot = Path(

        "images/eda/calibration_curve.png"

    )

    if calibration_plot.exists():

        st.image(

            str(calibration_plot),

            caption="Model Calibration Curve",

            use_container_width=True,

        )

    else:

        st.warning(

            "Calibration curve not found."

        )

    st.divider()

    # =====================================================
    # FEATURE IMPORTANCE
    # =====================================================

    st.subheader(

        "🔍 Important Predictive Features"

    )

    feature_plot = Path(

        "images/eda/xgboost_feature_importance.png"

    )

    if feature_plot.exists():

        st.image(

            str(feature_plot),

            caption="XGBoost Feature Importance",

            use_container_width=True,

        )

    else:

        st.warning(

            "Feature importance visualization not found."

        )