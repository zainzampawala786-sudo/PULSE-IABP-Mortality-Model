# ═══════════════════════════════════════════════════════════════════════════════
# PULSE-IABP RISK CALCULATOR
# Prediction Using Long-term Survival Estimation in AMI Patients Undergoing IABP Support
# ═══════════════════════════════════════════════════════════════════════════════
# Version: 1.0.0
# Developed by: Z. Zampawala et al. (2025)
# ═══════════════════════════════════════════════════════════════════════════════

import streamlit as st
import pickle
import numpy as np
import pandas as pd

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="PULSE-IABP Risk Calculator",
    page_icon="🫀",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ═══════════════════════════════════════════════════════════════════════════════
# CUSTOM CSS STYLING
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
    /* Main container */
    .main {
        max-width: 900px;
        margin: 0 auto;
        padding: 2rem;
    }
    
    /* Header */
    .header-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
    }
    
    .header-title {
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
    }
    
    .header-subtitle {
        font-size: 1rem;
        margin: 0.5rem 0 0 0;
        opacity: 0.95;
    }
    
    /* Section headers */
    .section-header {
        background-color: #f0f2f6;
        padding: 0.75rem 1rem;
        border-left: 4px solid #667eea;
        margin: 1.5rem 0 1rem 0;
        font-weight: 600;
        color: #2c3e50;
    }
    
    /* Result box */
    .result-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    .result-title {
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 1rem;
        opacity: 0.95;
    }
    
    .risk-score {
        font-size: 3.5rem;
        font-weight: 800;
        margin: 1rem 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .interpretation-text {
        font-size: 1.1rem;
        margin-top: 1.5rem;
        line-height: 1.6;
        opacity: 0.95;
    }
    
    /* Risk contributors box */
    .contributors-box {
        background-color: #fff3cd;
        border: 2px solid #ffc107;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 2rem 0;
    }
    
    .contributors-title {
        font-size: 1.2rem;
        font-weight: 700;
        color: #856404;
        margin-bottom: 1rem;
        text-align: center;
    }
    
    .contributor-item {
        background-color: white;
        padding: 0.75rem;
        margin: 0.5rem 0;
        border-radius: 5px;
        border-left: 4px solid #dc3545;
        font-size: 1rem;
        color: #212529;
    }
    
    /* Disclaimer */
    .disclaimer-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        padding: 1rem;
        border-radius: 8px;
        margin: 2rem 0;
        text-align: center;
    }
    
    .disclaimer-text {
        color: #721c24;
        font-size: 0.95rem;
        margin: 0;
    }
    
    /* Footer */
    .footer-text {
        text-align: center;
        padding: 1.5rem 0;
        border-top: 2px solid #e9ecef;
        color: #6c757d;
        font-size: 0.9rem;
        margin-top: 2rem;
    }
    
    /* Button styling */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-size: 1.1rem;
        font-weight: 600;
        padding: 0.75rem 2rem;
        border: none;
        border-radius: 8px;
        margin: 1.5rem 0;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    
    /* Radio button labels */
    .stRadio > label {
        font-weight: 500;
    }
    
    /* Slider width control */
    .stSlider {
        max-width: 100% !important;
    }
    
    /* Better spacing for columns */
    [data-testid="column"] {
        padding: 0 0.5rem;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main {
            max-width: 100%;
            padding: 1rem;
        }
        
        .header-title {
            font-size: 1.5rem;
        }
        
        .header-subtitle {
            font-size: 0.9rem;
        }
    }

</style>
""", unsafe_allow_html=True)
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# LOAD MODEL
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_model():
    try:
        with open("model_bundle.pkl", "rb") as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

bundle = load_model()
model = bundle["models"]["calibrated_svm"]
scaler = bundle["models"]["scaler"]
ref_risks = bundle["predictions"]["all_internal_calibrated"]
features = bundle["model_info"]["features"]

# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

# (NEW) Frozen probability thresholds defined on internal cohort
PROB_THRESHOLDS = {"low": 0.15, "medium": 0.35, "high": 0.60}

def calculate_risk_level(prob, ref):
    """Convert probability to percentile score (0-100)"""
    return (prob > ref).mean() * 100

def get_risk_category(level):
    """Get risk category and color"""
    if level < 25:
        return "LOW RISK", "#28a745"
    elif level < 50:
        return "MEDIUM RISK", "#ffc107"
    elif level < 75:
        return "ELEVATED RISK", "#fd7e14"
    else:
        return "CRITICAL RISK", "#dc3545"

# (NEW) Category by calibrated probability (for details box only)
def categorize_by_probability(p, thr=PROB_THRESHOLDS):
    if p < thr["low"]:
        return "LOW RISK"
    elif p < thr["medium"]:
        return "MEDIUM RISK"
    elif p < thr["high"]:
        return "ELEVATED RISK"
    else:
        return "CRITICAL RISK"

def get_risk_factors(inp):
    """Identify top 3 risk contributors"""
    factors = []
    
    if inp["lactate"] > 4.0:
        factors.append(f" Peak Lactate: {inp['lactate']:.1f} mmol/L (threshold >4.0)")
    
    if inp["age"] > 70:
        factors.append(f" Age: {inp['age']:.0f} years (threshold >70)")
    
    if inp["egfr"] < 45:
        factors.append(f" eGFR: {inp['egfr']:.0f} mL/min/1.73m² (threshold <45)")
    
    if inp["cpr"]:
        factors.append(" Cardiopulmonary Resuscitation: Performed")
    
    if inp["crrt"]:
        factors.append(" Continuous Renal Replacement: Required")
    
    if inp["vent"]:
        factors.append(" Invasive Mechanical Ventilation: Required")
    
    if inp["hgb_min"] < 90:
        factors.append(f" Minimum Hemoglobin: {inp['hgb_min']} g/L (threshold <90)")
    
    if inp["glucose_min"] < 5.5:
        factors.append(f" Minimum Glucose: {inp['glucose_min']:.1f} mmol/L (threshold <5.5)")
    
    return factors[:3]

# ═══════════════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="header-box">
    <div class="header-title">PULSE-IABP RISK CALCULATOR</div>
    <div class="header-subtitle">Prediction Using Long-term Survival Estimation in AMI Patients Undergoing IABP Support</div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PATIENT DEMOGRAPHICS
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="section-header">PATIENT DEMOGRAPHICS</div>', unsafe_allow_html=True)
age = st.slider("Age (years)", 18, 100, 65, key="age")

# ═══════════════════════════════════════════════════════════════════════════════
# PHARMACOTHERAPY
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="section-header">PHARMACOTHERAPY</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    beta_blocker = st.radio("β-Blocker Therapy", ["No", "Yes"], index=1, key="bb", horizontal=True)

with col2:
    ace_inhibitor = st.radio("ACE Inhibitor Therapy", ["No", "Yes"], index=0, key="acei", horizontal=True)

with col3:
    ticagrelor = st.radio("Ticagrelor Therapy", ["No", "Yes"], index=0, key="tica", horizontal=True)

# ═══════════════════════════════════════════════════════════════════════════════
# CRITICAL CARE INTERVENTIONS
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="section-header">CRITICAL CARE INTERVENTIONS</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    invasive_vent = st.radio("Invasive Mechanical Ventilation", ["No", "Yes"], index=0, key="vent", horizontal=True)

with col2:
    cpr = st.radio("Cardiopulmonary Resuscitation", ["No", "Yes"], index=0, key="cpr", horizontal=True)

with col3:
    crrt = st.radio("Continuous Renal Replacement", ["No", "Yes"], index=0, key="crrt", horizontal=True)

# ═══════════════════════════════════════════════════════════════════════════════
# HEMATOLOGY
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="section-header">HEMATOLOGY</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    hgb_min = st.slider("Hemoglobin, minimum (g/L)", 40, 180, 110, key="hgb_min")
    hgb_max = st.slider("Hemoglobin, peak (g/L)", 40, 180, 135, key="hgb_max")
    rbc_max = st.slider("RBC count, peak (×10¹²/L)", 2.0, 7.0, 4.5, 0.1, key="rbc")

with col2:
    neut_abs = st.slider("Neutrophils, minimum (×10⁹/L)", 0.0, 30.0, 5.0, 0.1, key="neut_abs")
    neut_pct = st.slider("Neutrophils, minimum (%)", 0, 100, 70, key="neut_pct")

# ═══════════════════════════════════════════════════════════════════════════════
# RENAL FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="section-header">RENAL FUNCTION</div>', unsafe_allow_html=True)

egfr = st.slider("eGFR CKD-EPI 2021 (mL/min/1.73m²)", 5, 120, 75, key="egfr")

# ═══════════════════════════════════════════════════════════════════════════════
# METABOLIC & ELECTROLYTES
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="section-header">METABOLIC & ELECTROLYTES</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    glucose_min = st.slider("Glucose, minimum (mmol/L)", 2.0, 25.0, 6.0, 0.1, key="glucose")

with col2:
    lactate_max = st.slider("Lactate, peak (mmol/L)", 0.0, 20.0, 2.5, 0.1, key="lactate")

with col3:
    sodium_max = st.slider("Sodium, peak (mmol/L)", 120, 160, 140, key="sodium")

# ═══════════════════════════════════════════════════════════════════════════════
# CALCULATE BUTTON
# ═══════════════════════════════════════════════════════════════════════════════

calc_btn = st.button("🔬 CALCULATE RISK SCORE")

# ═══════════════════════════════════════════════════════════════════════════════
# RESULTS SECTION
# ═══════════════════════════════════════════════════════════════════════════════

if calc_btn:
    # Map inputs to model features
    feat_map = {}
    
    for feature_name in features:
        if feature_name == "age":
            feat_map[feature_name] = age
        elif feature_name == "beta_blocker_use":
            feat_map[feature_name] = int(beta_blocker == "Yes")
        elif feature_name == "acei_use":
            feat_map[feature_name] = int(ace_inhibitor == "Yes")
        elif feature_name == "ticagrelor_use":
            feat_map[feature_name] = int(ticagrelor == "Yes")
        elif feature_name == "invasive_ventilation":
            feat_map[feature_name] = int(invasive_vent == "Yes")
        elif feature_name == "underwent_CPR":
            feat_map[feature_name] = int(cpr == "Yes")
        elif feature_name == "underwent_CRRT":
            feat_map[feature_name] = int(crrt == "Yes")
        elif feature_name == "hemoglobin_min":
            feat_map[feature_name] = hgb_min
        elif feature_name == "hemoglobin_max":
            feat_map[feature_name] = hgb_max
        elif feature_name == "rbc_count_max":
            feat_map[feature_name] = rbc_max
        elif feature_name == "neutrophils_abs_min":
            feat_map[feature_name] = neut_abs
        elif feature_name == "neutrophils_pct_min":
            feat_map[feature_name] = neut_pct
        elif feature_name == "eGFR_CKD_EPI_21":
            feat_map[feature_name] = egfr
        elif feature_name == "glucose_min":
            feat_map[feature_name] = glucose_min
        elif feature_name == "lactate_max":
            feat_map[feature_name] = lactate_max
        elif feature_name == "sodium_max":
            feat_map[feature_name] = sodium_max
        else:
            feat_map[feature_name] = 0
    
    # Create input array and predict
    X = np.array([[feat_map[f] for f in features]])
    X_scaled = scaler.transform(X)
    prob = model.predict_proba(X_scaled)[0, 1]
    risk_level = calculate_risk_level(prob, ref_risks)
    category, color = get_risk_category(risk_level)
    
    # Display results
    st.markdown(f"""
    <div class="result-container">
        <div class="result-title">ONE-YEAR MORTALITY RISK ASSESSMENT</div>
        <div class="risk-score">RISK SCORE: {risk_level:.0f}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Progress bar
    st.markdown(f"""
    <div style="margin: 2rem 0;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem; font-size: 0.9rem; color: #6c757d;">
            <span>0</span>
            <span>25</span>
            <span>50</span>
            <span>75</span>
            <span>100</span>
        </div>
        <div style="width: 100%; height: 40px; background-color: #e9ecef; border-radius: 20px; overflow: hidden; position: relative;">
            <div style="width: {risk_level}%; height: 100%; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); transition: width 0.5s ease;"></div>
            <div style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; display: flex; justify-content: space-between; align-items: center; padding: 0 1rem; font-size: 0.8rem; font-weight: 600; color: #495057;">
                <span>LOW (0-24)</span>
                <span>MEDIUM (25-49)</span>
                <span>ELEVATED (50-74)</span>
                <span>CRITICAL (75-100)</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div style="text-align: center; font-size: 1.1rem; color: #495057; margin: 1.5rem 0;">
        Patient’s estimated one-year mortality risk is higher than <strong>{risk_level:.0f}%</strong> of patients with AMI treated with IABP support.
    </div>
    """, unsafe_allow_html=True)
    
    # Risk contributors
    inp_dict = {
        "age": age,
        "egfr": egfr,
        "lactate": lactate_max,
        "cpr": (cpr == "Yes"),
        "crrt": (crrt == "Yes"),
        "vent": (invasive_vent == "Yes"),
        "hgb_min": hgb_min,
        "glucose_min": glucose_min
    }
    
    factors = get_risk_factors(inp_dict)
    
    if factors:
        st.markdown('<div class="contributors-box">', unsafe_allow_html=True)
        st.markdown('<div class="contributors-title">PRIMARY RISK CONTRIBUTORS</div>', unsafe_allow_html=True)
        
        for factor in factors:
            st.markdown(f'<div class="contributor-item">{factor}</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────────────
    # DETAILS (research-only): model probability & thresholds  (appears after Calc)
    # ─────────────────────────────────────────────────────────────────────────────
    with st.expander("Model derivation and scoring methodology (research only)"):
        prob_pct = float(prob * 100.0)
        cat_prob = categorize_by_probability(prob)

        st.write(
    "**Display score (0–100)** represents the percentile rank of this patient’s "
    "**calibrated mortality probability** relative to the internal (Tongji Hospital) reference cohort. "
    "For example, a score of 80 indicates a higher predicted risk than 80% of comparable AMI patients requiring IABP support."
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Calibrated probability", f"{prob_pct:.1f}%")
        with c2:
            st.metric("Percentile score", f"{risk_level:.0f}")
        with c3:
            st.metric("Probability tier", cat_prob)

        st.markdown("**Frozen probability thresholds (defined on internal cohort):**")
        th = PROB_THRESHOLDS
        st.markdown(
            f"- **LOW:** p < **{th['low']:.2f}**  \n"
            f"- **MEDIUM:** **{th['low']:.2f} ≤ p < {th['medium']:.2f}**  \n"
            f"- **ELEVATED:** **{th['medium']:.2f} ≤ p < {th['high']:.2f}**  \n"
            f"- **CRITICAL:** p ≥ **{th['high']:.2f}**"
        )

        st.caption(
    "The on-screen 0–100 score is a percentile rank for intuitive interpretation. "
    "Probability thresholds are based on calibrated outputs and fixed from the internal cohort."
        )

# ═══════════════════════════════════════════════════════════════════════════════
# DISCLAIMER
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="disclaimer-box">
    <div class="disclaimer-text">
        <strong>️ DISCLAIMER:</strong> This calculator is for RESEARCH and EDUCATIONAL purposes only. NOT validated for clinical decision-making.
    </div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="footer-text">
    IABP-PULSE Calculator v1.0.0 • Streamlit • Developed by Z. Zampawala et al. (2025)
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR - MODEL INFORMATION
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### ℹ️ MODEL INFORMATION")
    
    st.markdown("**ALGORITHM SPECIFICATIONS:**")
    st.markdown("• Model: Support Vector Machine (RBF kernel)")
    st.markdown("• Calibration: Platt scaling (sigmoid)")
    st.markdown("• Hyperparameters: C=0.1, γ=auto, class_weight=balanced")
    st.markdown("• Feature Selection: 16 clinical variables")
    st.markdown("• Missing Data: Median imputation")
    
    st.markdown("**STUDY POPULATION:**")
    st.markdown("• Training Cohort: n=476 (Tongji Hospital, Wuhan, China)")
    st.markdown("• Validation Cohort: n=354 (MIMIC-IV, Boston, USA)")
    st.markdown("• Inclusion: AMI patients undergoing IABP")
    st.markdown("• Follow-up: 12 months")
    
    st.markdown("**PERFORMANCE METRICS:**")
    st.markdown("• External AUC: 0.768 (95% CI: 0.72-0.82)")
    st.markdown("• Brier Score: 0.186")
    st.markdown("• Calibration Slope: 0.918 (SE: 0.08)")
    
    st.markdown("---")
    
    st.markdown("### 📄 CITATION INFORMATION")
    st.markdown("""
    Zampawala Z, et al. (2025). PULSE-IABP: Machine Learning Risk Calculator 
    for One-Year Mortality in AMI Patients with IABP Support. 
    [Journal Name]. [In Press].
    
    DOI: [To be assigned]
    
     2025 Z. Zampawala et al. All rights reserved.
    """)






