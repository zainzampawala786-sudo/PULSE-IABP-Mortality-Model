# PULSE-IABP Risk Calculator

**One-Year Mortality Risk Assessment for AMI Patients with IABP Support**

Version: 1.0.0
Date: 2025-10-17
Author: Z. Zampawala et al. 

---

## Quick Start

```bash
pip install -r requirements.txt
streamlit run pulse_iabp_calculator.py
```

Open browser: http://localhost:8501

---

## Deploy to Streamlit Cloud

1. Push to GitHub
2. Go to share.streamlit.io
3. Connect repository
4. Set main file: pulse_iabp_calculator.py
5. Deploy

---

## Model Information

- Training: n=476 (internal cohort only)
- Validation: n=354 (external cohort)
- External AUC: 0.768
- Display: Risk Score 0-100 (probability-based)

---

## Risk Stratification Thresholds
- LOW: < 15% (Internal: 5.1% mortality)
- MEDIUM: 15-45% (Internal: 22.3% mortality)
- HIGH: 45-70% (Internal: 45.3% mortality)
- VERY HIGH: ≥ 70% (Internal: 91.1% mortality)

## Features

- Risk Score: 0-100 (probability × 100 = mortality risk)
- Categories: LOW / MEDIUM / HIGH / VERY HIGH
- Units included for all variables
- Professional medical interface

---

## Disclaimer

WARNING: For research and educational purposes only.
NOT approved for clinical decision-making.


