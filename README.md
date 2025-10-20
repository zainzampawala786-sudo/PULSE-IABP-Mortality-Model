# PULSE-IABP Risk Calculator

**One-Year Mortality Risk Assessment for AMI Patients with IABP Support**

Version: 1.0.0
Date: 2025-10-20
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
- Display: Risk Level 0-100 (percentile-based)
- TRIPOD Type: 3 (External validation)

---

## Features

- Risk Level: 0-100 (percentile score)
- Categories: LOW / MEDIUM / ELEVATED / CRITICAL
- Top 3 risk factors displayed
- Units included for all variables
- Professional medical interface

---

## Disclaimer

WARNING: For research and educational purposes only.
NOT approved for clinical decision-making.

