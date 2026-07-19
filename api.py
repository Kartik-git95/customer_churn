from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import io
import base64

app = FastAPI()

# Allow React (Bolt.new) to talk to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, put your Vercel URL here
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
model = joblib.load("model.pkl")
feature_names = joblib.load("feature_names.pkl")
explainer = shap.TreeExplainer(model)

# Copy your EXACT build_feature_row function from app.py and paste it here
def build_feature_row(inp: dict) -> pd.DataFrame:
    d = {}
    d["Age"] = inp["age"]
    # ... (paste the rest of the function exactly as it is in app.py) ...
    # ... I am truncating it here for brevity ...
    row = pd.DataFrame([d])
    missing = set(feature_names) - set(row.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    return row[feature_names]

@app.post("/predict")
def predict_churn(data: dict):
    try:
        # 1. Build features and predict
        X_input = build_feature_row(data)
        proba = float(model.predict_proba(X_input)[0, 1])
        
        # 2. Generate SHAP plot
        shap_values = explainer(X_input)
        fig = plt.figure()
        shap.plots.waterfall(shap_values[0], max_display=12, show=False)
        
        # 3. Convert plot to Base64 image so React can display it
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches='tight', dpi=150)
        buf.seek(0)
        plot_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        
        # 4. Send back JSON response
        return {
            "probability": proba,
            "risk_level": "High" if proba >= 0.5 else "Medium" if proba >= 0.25 else "Low",
            "shap_plot_image": f"data:image/png;base64,{plot_base64}"
        }
    except Exception as e:
        return {"error": str(e)}