# Streamlit Cloud Deployment

## Step 1: Push to GitHub

```bash
cd "C:\Users\zainz\Desktop\Second Analysis\ZAINY\models\mortalitybundlecalculator"
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/pulse-iabp.git
git push -u origin main
```

## Step 2: Deploy

1. Go to: share.streamlit.io
2. Sign in with GitHub
3. Click New app
4. Select repository
5. Main file: pulse_iabp_calculator.py
6. Deploy
