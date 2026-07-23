import streamlit as st

from src.dashboard.styles import CUSTOM_CSS

from src.dashboard.sidebar import show_sidebar

from src.dashboard.prediction_tab import (
    show_prediction_tab,
)

from src.dashboard.model_insights import (
    show_model_insights,
)

from src.dashboard.about_tab import (
    show_about_tab,
)


# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(

    page_title="AI Clinical Decision Support",

    page_icon="🏥",

    layout="wide",

)


# =====================================================
# CUSTOM STYLES
# =====================================================

st.markdown(

    CUSTOM_CSS,

    unsafe_allow_html=True,

)


# =====================================================
# SIDEBAR
# =====================================================

show_sidebar()


# =====================================================
# DASHBOARD TABS
# =====================================================

prediction_tab, insights_tab, about_tab = st.tabs(

    [

        "🏥 Risk Prediction",

        "📊 Model Insights",

        "ℹ️ About the System",

    ]

)


# =====================================================
# RISK PREDICTION
# =====================================================

with prediction_tab:

    show_prediction_tab()


# =====================================================
# MODEL INSIGHTS
# =====================================================

with insights_tab:

    show_model_insights()


# =====================================================
# ABOUT
# =====================================================

with about_tab:

    show_about_tab()


# =====================================================
# FOOTER
# =====================================================

st.divider()

st.caption(

    "AI-Assisted Clinical Decision Support System • "
    "Python • XGBoost • Streamlit"

)