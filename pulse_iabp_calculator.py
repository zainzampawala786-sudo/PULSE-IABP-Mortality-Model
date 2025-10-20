# PULSE-IABP Risk Calculator
# Version: 1.0.0 | Date: 2025-01-20
# Training: n=476 (internal only) | External AUC: 0.768
# FIX: Correct feature names (acei_use, underwent_CRRT)

import streamlit as st
import pickle
import numpy as np
import pandas as pd

st.set_page_config(
    page_title="PULSE-IABP Risk Calculator",
    page_icon=":heart:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load model
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

def calculate_risk_level(prob, ref):
    return (prob > ref).mean() * 100

def get_risk_category(level):
    if level < 25:
        return "LOW RISK", ":green_circle:"
    elif level < 50:
        return "MEDIUM RISK", ":yellow_circle:"
    elif level < 75:
        return "ELEVATED RISK", ":orange_circle:"
    return "CRITICAL RISK", ":red_circle:"

def get_risk_factors(inp):
    factors = []
    if inp["lactate"] > 4.0:
        factors.append(f"Peak Lactate ({inp['lactate']:.1f} mmol/L) - Elevated")
    if inp["age"] > 70:
        factors.append(f"Age ({inp['age']:.0f} years) - Advanced")
    if inp["egfr"] < 45:
        factors.append(f"eGFR ({inp['egfr']:.0f} mL/min/1.73m2) - Impaired")
    if inp["cpr"]:
        factors.append("CPR performed")
    if inp["crrt"]:
        factors.append("CRRT required")
    if inp["vent"]:
        factors.append("Invasive ventilation")
    if inp["hgb_min"] < 90:
        factors.append(f"Minimum Hemoglobin ({inp['hgb_min']} g/L) - Severely low")
    return factors[:3]

# Header
st.title(":heart: PULSE-IABP Risk Calculator")
st.caption("One-Year Mortality Risk Assessment for AMI Patients with IABP Support")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("Patient Information")
    st.markdown("---")
    st.subheader("Demographics")
    age = st.slider("Age (years)", 18, 100, 65)
    st.markdown("---")
    st.subheader("Medications")
    beta_blocker = st.checkbox("Beta-Blocker")
    ace_inhibitor = st.checkbox("ACE Inhibitor")
    ticagrelor = st.checkbox("Ticagrelor")
    st.markdown("---")
    st.subheader("Interventions")
    invasive_vent = st.checkbox("Invasive Ventilation")
    cpr = st.checkbox("CPR Performed")
    crrt = st.checkbox("CRRT")
    st.markdown("---")
    st.subheader("Laboratory Values")
    st.markdown("**Hematology**")
    hgb_min = st.slider("Min Hemoglobin (g/L)", 40, 180, 110)
    hgb_max = st.slider("Peak Hemoglobin (g/L)", 40, 180, 135)
    rbc_max = st.slider("Peak RBC (x10^12/L)", 2.0, 7.0, 4.5, 0.1)
    neut_abs = st.slider("Min Neutrophils (x10^9/L)", 0.0, 30.0, 5.0, 0.1)
    neut_pct = st.slider("Min Neutrophils (%)", 0, 100, 70)
    st.markdown("**Renal Function**")
    egfr = st.slider("eGFR (mL/min/1.73m2)", 5, 120, 75)
    st.markdown("**Metabolic**")
    glucose_min = st.slider("Min Glucose (mmol/L)", 2.0, 25.0, 6.0, 0.1)
    lactate_max = st.slider("Peak Lactate (mmol/L)", 0.0, 20.0, 2.5, 0.1)
    sodium_max = st.slider("Peak Sodium (mmol/L)", 120, 160, 140)
    st.markdown("---")
    calc_btn = st.button("CALCULATE RISK", type="primary", use_container_width=True)

# Main
if not calc_btn:
    st.info("Enter patient information in sidebar and click CALCULATE RISK")
    with st.expander("About"):
        st.write("Training: n=476 (internal) | Validation: n=354 (external) | AUC: 0.768")
else:
    try:
        # CORRECTED FEATURE MAPPING
        feat_map = {}
        
        for feature_name in features:
            if feature_name == "age":
                feat_map[feature_name] = age
            elif feature_name == "beta_blocker_use":
                feat_map[feature_name] = int(beta_blocker)
            elif feature_name == "acei_use":  # ← CORRECTED
                feat_map[feature_name] = int(ace_inhibitor)
            elif feature_name == "ticagrelor_use":
                feat_map[feature_name] = int(ticagrelor)
            elif feature_name == "invasive_ventilation":
                feat_map[feature_name] = int(invasive_vent)
            elif feature_name == "underwent_CPR":
                feat_map[feature_name] = int(cpr)
            elif feature_name == "underwent_CRRT":  # ← CORRECTED
                feat_map[feature_name] = int(crrt)
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
                st.warning(f"Unknown feature: {feature_name}")
        
        X = np.array([[feat_map[f] for f in features]])
        X_scaled = scaler.transform(X)
        prob = model.predict_proba(X_scaled)[0, 1]
        risk_level = calculate_risk_level(prob, ref_risks)
        category, emoji = get_risk_category(risk_level)
        
        st.markdown(f"### {emoji} PULSE-IABP Risk Level: **{risk_level:.0f}**")
        st.markdown(f"### Category: **{category}**")
        st.progress(risk_level / 100)
        st.info(f"Higher risk than {risk_level:.0f}% of similar patients with AMI requiring IABP support.")
        
        st.markdown("---")
        
        inp = {
            "age": age, 
            "egfr": egfr, 
            "lactate": lactate_max, 
            "cpr": cpr, 
            "crrt": crrt, 
            "vent": invasive_vent,
            "hgb_min": hgb_min
        }
        factors = get_risk_factors(inp)
        
        if factors:
            st.markdown("#### Key Risk Factors")
            for fac in factors:
                st.markdown(f"- {fac}")
        
        st.markdown("---")
        st.markdown("#### Risk Categories")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("LOW", "0-24")
        c2.metric("MEDIUM", "25-49")
        c3.metric("ELEVATED", "50-74")
        c4.metric("CRITICAL", "75-100")
        
    except Exception as e:
        st.error(f"Error calculating risk: {e}")

st.markdown("---")
st.warning("DISCLAIMER: For research and educational purposes only. NOT for clinical decision-making.")
st.caption("Model: SVM-RBF + Platt | Training: n=476 | External AUC: 0.768 | Version: 1.0.2")
