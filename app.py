import streamlit as st
import joblib
import pandas as pd

# --------------------------------
# PAGE CONFIG
# --------------------------------
st.set_page_config(
    page_title="AquaCheck | Water Predictor",
    page_icon="💧",
    layout="wide"
)

# --------------------------------
# LOAD MODEL
# --------------------------------
@st.cache_resource
def load_model():
    return joblib.load("water_model.pkl")

model = load_model()

FEATURES = [
    "ph",
    "Hardness",
    "Solids",
    "Chloramines",
    "Sulfate",
    "Conductivity",
    "Organic_carbon",
    "Trihalomethanes",
    "Turbidity"
]

# --------------------------------
# CUSTOM CSS  (Deep Ocean theme)
# --------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'Poppins', sans-serif;
}

.stApp {
    background: radial-gradient(circle at top left, #0F3B4C 0%, #071A24 55%, #04111A 100%);
    color: #E6F4F1;
}

header[data-testid="stHeader"] {
    background-color: transparent;
}

.block-container {
    padding-top: 2rem;
    max-width: 1150px;
}

/* Hero banner */
.hero {
    background: linear-gradient(135deg, #14B8A6 0%, #0E7490 50%, #1E3A8A 100%);
    border-radius: 22px;
    padding: 38px 30px;
    text-align: center;
    box-shadow: 0 18px 40px rgba(20, 184, 166, 0.25);
    margin-bottom: 28px;
}

.hero-title {
    font-size: 46px;
    font-weight: 800;
    color: #FFFFFF;
    letter-spacing: 1px;
    margin: 0;
}

.hero-sub {
    color: #CCFBF1;
    font-size: 17px;
    font-weight: 500;
    margin-top: 6px;
}

.hero-text {
    color: #E0F2FE;
    font-size: 15px;
    margin-top: 14px;
    opacity: 0.9;
}

/* Section headings */
.section-title {
    font-size: 22px;
    font-weight: 700;
    color: #5EEAD4;
    margin: 10px 0 4px 0;
}

.section-caption {
    color: #94A3B8;
    font-size: 14px;
    margin-bottom: 16px;
}

/* Group cards around each column */
div[data-testid="stColumn"] > div {
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(94, 234, 212, 0.18);
    border-radius: 18px;
    padding: 18px 16px;
    backdrop-filter: blur(6px);
}

.group-label {
    font-size: 13px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #2DD4BF;
    margin-bottom: 6px;
}

/* Input labels */
.stApp label,
.stApp [data-testid="stWidgetLabel"] p {
    color: #E2E8F0 !important;
    font-weight: 500 !important;
}

/* Input fields */
.stApp div[data-baseweb="input"] {
    background-color: #0B2530 !important;
    border: 1px solid #1F4E5F !important;
    border-radius: 12px !important;
}

.stApp div[data-baseweb="input"] input {
    color: #F0FDFA !important;
    background-color: #0B2530 !important;
    -webkit-text-fill-color: #F0FDFA !important;
}

.stApp div[data-baseweb="input"]:focus-within {
    border: 1px solid #2DD4BF !important;
    box-shadow: 0 0 0 3px rgba(45, 212, 191, 0.2) !important;
}

.stApp div[data-baseweb="input"] button {
    background-color: #12333F !important;
    color: #5EEAD4 !important;
}

/* Button */
.stButton > button {
    background: linear-gradient(90deg, #14B8A6, #0EA5E9) !important;
    color: #FFFFFF !important;
    border-radius: 999px !important;
    border: none !important;
    height: 54px;
    font-size: 17px !important;
    font-weight: 700 !important;
    letter-spacing: 0.5px;
    box-shadow: 0 10px 25px rgba(14, 165, 233, 0.3);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 14px 30px rgba(20, 184, 166, 0.45);
}

/* Result cards */
.result-card {
    border-radius: 20px;
    padding: 28px;
    text-align: center;
    margin: 10px 0 18px 0;
}

.result-good {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.25), rgba(20, 184, 166, 0.12));
    border: 1px solid #10B981;
}

.result-bad {
    background: linear-gradient(135deg, rgba(244, 63, 94, 0.25), rgba(249, 115, 22, 0.12));
    border: 1px solid #F43F5E;
}

.result-icon {
    font-size: 52px;
}

.result-title {
    font-size: 30px;
    font-weight: 800;
    margin: 6px 0;
}

.result-good .result-title { color: #6EE7B7; }
.result-bad  .result-title { color: #FDA4AF; }

.result-text {
    color: #CBD5E1;
    font-size: 15px;
}

/* Expanders */
div[data-testid="stExpander"] {
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(94, 234, 212, 0.18) !important;
    border-radius: 14px !important;
}

div[data-testid="stExpander"] summary,
div[data-testid="stExpander"] p,
div[data-testid="stExpander"] li {
    color: #E2E8F0 !important;
}

/* Captions / dividers */
.stApp [data-testid="stCaptionContainer"] {
    color: #94A3B8 !important;
}

hr {
    border-color: rgba(94, 234, 212, 0.15) !important;
}

/* Footer */
.footer {
    text-align: center;
    color: #64748B;
    font-size: 13px;
    margin-top: 30px;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------
# HEADER
# --------------------------------
st.markdown(
    """
    <div class="hero">
        <div class="hero-title">💧 AquaCheck</div>
        <div class="hero-sub">Water Quality &amp; Potability Prediction System</div>
        <div class="hero-text">
            Enter water quality measurements to predict whether a water
            sample belongs to the potable or non-potable category.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# --------------------------------
# INPUT SECTION
# --------------------------------
st.markdown(
    '<div class="section-title">🔬 Water Quality Parameters</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-caption">'
    'Suggested starting values are provided below. '
    'You can edit every field according to your sample.'
    '</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3, gap="medium")

with col1:

    st.markdown('<div class="group-label">⚗️ Chemistry</div>', unsafe_allow_html=True)

    ph = st.number_input(
        "pH Value",
        min_value=0.0,
        max_value=14.0,
        value=7.0,
        step=0.1,
        help="Acidity or alkalinity of water."
    )

    hardness = st.number_input(
        "Hardness",
        min_value=0.0,
        value=200.0,
        help="Calcium and magnesium content."
    )

    solids = st.number_input(
        "Total Dissolved Solids",
        min_value=0.0,
        value=20000.0,
        help="Dissolved minerals and salts."
    )

with col2:

    st.markdown('<div class="group-label">🧪 Treatment</div>', unsafe_allow_html=True)

    chloramines = st.number_input(
        "Chloramines",
        min_value=0.0,
        value=7.0,
        help="Disinfectant compounds in water."
    )

    sulfate = st.number_input(
        "Sulfate",
        min_value=0.0,
        value=300.0,
        help="Sulfate concentration."
    )

    conductivity = st.number_input(
        "Conductivity",
        min_value=0.0,
        value=400.0,
        help="Ability of water to conduct electricity."
    )

with col3:

    st.markdown('<div class="group-label">🌿 Organic &amp; Clarity</div>', unsafe_allow_html=True)

    organic_carbon = st.number_input(
        "Organic Carbon",
        min_value=0.0,
        value=10.0,
        help="Organic carbon content."
    )

    trihalomethanes = st.number_input(
        "Trihalomethanes",
        min_value=0.0,
        value=60.0,
        help="Disinfection by-products."
    )

    turbidity = st.number_input(
        "Turbidity",
        min_value=0.0,
        value=4.0,
        help="Cloudiness of water."
    )

st.write("")

# --------------------------------
# PREDICTION BUTTON
# --------------------------------
if st.button(
    "🔍 Analyze Water Sample",
    type="primary",
    use_container_width=True
):

    # Create fresh input on every click
    input_data = pd.DataFrame([{
        "ph": ph,
        "Hardness": hardness,
        "Solids": solids,
        "Chloramines": chloramines,
        "Sulfate": sulfate,
        "Conductivity": conductivity,
        "Organic_carbon": organic_carbon,
        "Trihalomethanes": trihalomethanes,
        "Turbidity": turbidity
    }], columns=FEATURES)

    try:

        # Match model's training feature order
        if hasattr(model, "feature_names_in_"):
            input_data = input_data.loc[
                :, model.feature_names_in_
            ]

        # Make a fresh prediction
        prediction = model.predict(input_data)[0]

        st.markdown(
            '<div class="section-title">📊 Prediction Result</div>',
            unsafe_allow_html=True
        )

        if prediction == 1:

            st.markdown(
                """
                <div class="result-card result-good">
                    <div class="result-icon">✅</div>
                    <div class="result-title">Water is Potable!</div>
                    <div class="result-text">
                        The model predicts that this sample belongs to the potable class.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                """
                <div class="result-card result-bad">
                    <div class="result-icon">⚠️</div>
                    <div class="result-title">Water is Not Potable!</div>
                    <div class="result-text">
                        The model predicts that this sample belongs to the non-potable class.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # Show latest input values
        with st.expander("View Current Input Values"):

            st.dataframe(
                input_data,
                use_container_width=True,
                hide_index=True
            )

        st.caption(
            "This is a machine learning prediction, "
            "not a laboratory water safety test."
        )

    except Exception as e:

        st.error(f"Prediction Error: {e}")

# --------------------------------
# ABOUT PROJECT
# --------------------------------
st.write("")

with st.expander("ℹ️ About AquaCheck"):

    st.write("""
    AquaCheck uses a Decision Tree Classifier
    to predict water potability.

    Dataset: Water Potability Dataset

    Dataset Samples: 3,276

    Input Features: 9 water quality parameters

    Target:
    - 1 = Potable
    - 0 = Not Potable

    Technology:
    Python, Pandas, Scikit-learn and Streamlit.
    """)

st.markdown(
    '<div class="footer">Built with Python · Scikit-learn · Streamlit</div>',
    unsafe_allow_html=True
)
