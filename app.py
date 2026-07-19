"""
Customer Churn Predictor - Streamlit App
Loads a trained XGBoost model + SHAP explainer, takes raw customer
inputs from a form, encodes them to match the training feature set,
and shows the churn probability with a per-prediction SHAP waterfall.

Run locally with:  streamlit run app.py

Required files in the same folder:
  - model.pkl            (trained XGBoost model)
  - feature_names.pkl    (exact training column order)
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import shap
import joblib

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="wide")


# ---------------------------------------------------------------
# Load model + explainer once, cache across reruns
# ---------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("model.pkl")
    feature_names = joblib.load("feature_names.pkl")
    explainer = shap.TreeExplainer(model)
    return model, explainer, feature_names


try:
    model, explainer, FEATURE_NAMES = load_artifacts()
except FileNotFoundError as e:
    st.error(
        "Missing model files. Ensure model.pkl and feature_names.pkl "
        f"are in this app's folder.\n\nDetails: {e}"
    )
    st.stop()


# ---------------------------------------------------------------
# Encode raw form inputs into the exact feature row the model expects
# ---------------------------------------------------------------
def build_feature_row(inp: dict) -> pd.DataFrame:
    d = {}

    d["Age"] = inp["age"]
    d["Married"] = int(inp["married"])
    d["Dependents"] = int(inp["num_dependents"] > 0)
    d["Number_of_Dependents"] = inp["num_dependents"]
    d["Population"] = inp["population"]
    d["Referred_a_Friend"] = int(inp["referred"])
    d["Number_of_Referrals"] = inp["num_referrals"]
    d["Tenure_in_Months"] = inp["tenure"]
    d["Phone_Service"] = int(inp["phone_service"])
    d["Avg_Monthly_Long_Distance_Charges"] = inp["avg_ld_charge"]
    d["Multiple_Lines"] = int(inp["multiple_lines"])
    d["Internet_Service"] = int(inp["internet_service"])
    d["Avg_Monthly_GB_Download"] = inp["avg_gb"]
    d["Online_Security"] = int(inp["online_security"])
    d["Online_Backup"] = int(inp["online_backup"])
    d["Device_Protection_Plan"] = int(inp["device_protection"])
    d["Premium_Tech_Support"] = int(inp["premium_support"])
    d["Streaming_TV"] = int(inp["streaming_tv"])
    d["Streaming_Movies"] = int(inp["streaming_movies"])
    d["Streaming_Music"] = int(inp["streaming_music"])
    d["Unlimited_Data"] = int(inp["unlimited_data"])
    d["Paperless_Billing"] = int(inp["paperless"])
    d["Monthly_Charge"] = inp["monthly_charge"]
    d["Total_Charges"] = inp["total_charges"]
    d["Total_Refunds"] = inp["total_refunds"]
    d["Total_Extra_Data_Charges"] = inp["total_extra_data"]
    d["Total_Long_Distance_Charges"] = inp["total_ld_charges"]
    d["Total_Revenue"] = inp["total_charges"] + inp["total_ld_charges"] - inp["total_refunds"]
    d["CLTV"] = inp["cltv"]

    safe_tenure = max(inp["tenure"], 1)
    d["AvgChargesPerMonth"] = inp["total_charges"] / safe_tenure
    d["HasSecurityBundle"] = int(inp["online_security"] and inp["premium_support"])
    d["TotalServicesSubscribed"] = sum([
        inp["online_security"], inp["online_backup"], inp["device_protection"],
        inp["premium_support"], inp["streaming_tv"], inp["streaming_movies"],
        inp["streaming_music"],
    ])
    d["HasRefund"] = int(inp["total_refunds"] > 0)

    d["Gender_Male"] = inp["gender"] == "Male"

    for offer_name in ["Offer A", "Offer B", "Offer C", "Offer D", "Offer E"]:
        d[f"Offer_{offer_name}"] = inp["offer"] == offer_name

    for net_type in ["DSL", "Fiber Optic", "No Internet Service"]:
        d[f"Internet_Type_{net_type}"] = inp["internet_type"] == net_type

    d["Contract_One Year"] = inp["contract"] == "One Year"
    d["Contract_Two Year"] = inp["contract"] == "Two Year"

    d["Payment_Method_Credit Card"] = inp["payment_method"] == "Credit Card"
    d["Payment_Method_Mailed Check"] = inp["payment_method"] == "Mailed Check"

    t = inp["tenure"]
    d["TenureGroup_1-2 Years"] = 13 <= t <= 24
    d["TenureGroup_2-4 Years"] = 25 <= t <= 48
    d["TenureGroup_4+ Years"] = t >= 49

    row = pd.DataFrame([d])

    missing = set(FEATURE_NAMES) - set(row.columns)
    extra = set(row.columns) - set(FEATURE_NAMES)
    if missing:
        st.error(f"App is missing columns the model expects: {sorted(missing)}")
        st.stop()
    if extra:
        row = row.drop(columns=list(extra))

    return row[FEATURE_NAMES]


# ---------------------------------------------------------------
# UI
# ---------------------------------------------------------------
st.title("📉 Customer Churn Predictor")
st.caption(
    "Trained XGBoost model on the IBM Telco Customer Churn dataset. "
    "Fill in a customer profile to see their churn risk and what's driving it."
)

app_mode = st.radio("Choose Mode", ["Single Customer", "Batch CSV Upload"], horizontal=True)


# ---------------------------------------------------------------
# MODE 1: Single Customer Form (Local XAI)
# ---------------------------------------------------------------
if app_mode == "Single Customer":
    with st.sidebar.form("customer_form"):
        
        with st.expander("👤 Demographics", expanded=True):
            age = st.number_input("Age", 18, 100, 45)
            gender = st.radio("Gender", ["Female", "Male"], horizontal=True)
            married = st.checkbox("Married")
            num_dependents = st.number_input("Number of Dependents", 0, 10, 0)
            population = st.number_input("ZIP Code Population", 0, 200000, 20000, step=1000)

        with st.expander("📅 Account Details", expanded=True):
            tenure = st.number_input("Tenure (months)", 0, 100, 12)
            contract = st.selectbox("Contract", ["Month-to-Month", "One Year", "Two Year"])
            payment_method = st.selectbox("Payment Method", ["Bank Withdrawal", "Credit Card", "Mailed Check"])
            paperless = st.checkbox("Paperless Billing", value=True)
            offer = st.selectbox("Current Offer", ["None", "Offer A", "Offer B", "Offer C", "Offer D", "Offer E"])
            referred = st.checkbox("Referred a Friend")
            num_referrals = st.number_input("Number of Referrals", 0, 20, 0)

        with st.expander("📱 Services Subscribed"):
            phone_service = st.checkbox("Phone Service", value=True)
            multiple_lines = st.checkbox("Multiple Lines")
            internet_service = st.checkbox("Internet Service", value=True)
            internet_type = st.selectbox("Internet Type", ["Cable", "DSL", "Fiber Optic", "No Internet Service"])
            avg_gb = st.number_input("Avg Monthly GB Download", 0, 200, 20)
            online_security = st.checkbox("Online Security")
            online_backup = st.checkbox("Online Backup")
            device_protection = st.checkbox("Device Protection Plan")
            premium_support = st.checkbox("Premium Tech Support")
            streaming_tv = st.checkbox("Streaming TV")
            streaming_movies = st.checkbox("Streaming Movies")
            streaming_music = st.checkbox("Streaming Music")
            unlimited_data = st.checkbox("Unlimited Data", value=True)

        with st.expander("💰 Billing & Charges"):
            monthly_charge = st.number_input("Monthly Charge ($)", 0.0, 500.0, 65.0, step=0.5)
            avg_ld_charge = st.number_input("Avg Monthly Long Distance Charges ($)", 0.0, 100.0, 5.0, step=0.5)
            total_charges = st.number_input("Total Charges ($)", 0.0, 20000.0, float(monthly_charge * max(tenure, 1)), step=1.0)
            total_ld_charges = st.number_input("Total Long Distance Charges ($)", 0.0, 5000.0, float(avg_ld_charge * max(tenure, 1)), step=1.0)
            total_refunds = st.number_input("Total Refunds ($)", 0.0, 2000.0, 0.0, step=1.0)
            total_extra_data = st.number_input("Total Extra Data Charges ($)", 0.0, 1000.0, 0.0, step=1.0)
            cltv = st.number_input("CLTV Score", 0, 10000, 4500)

        submitted = st.form_submit_button("Predict Churn Risk", use_container_width=True)

    if submitted:
        inputs = dict(
            age=age, married=married, num_dependents=num_dependents, population=population,
            referred=referred, num_referrals=num_referrals, tenure=tenure,
            phone_service=phone_service, avg_ld_charge=avg_ld_charge, multiple_lines=multiple_lines,
            internet_service=internet_service, avg_gb=avg_gb, online_security=online_security,
            online_backup=online_backup, device_protection=device_protection,
            premium_support=premium_support, streaming_tv=streaming_tv, streaming_movies=streaming_movies,
            streaming_music=streaming_music, unlimited_data=unlimited_data, paperless=paperless,
            monthly_charge=monthly_charge, total_charges=total_charges, total_refunds=total_refunds,
            total_extra_data=total_extra_data, total_ld_charges=total_ld_charges, cltv=cltv,
            gender=gender, offer=offer, internet_type=internet_type, contract=contract,
            payment_method=payment_method,
        )

        X_input = build_feature_row(inputs)
        proba = model.predict_proba(X_input)[0, 1]

        col1, col2 = st.columns([1, 2])

        with col1:
            st.metric("Predicted Churn Probability", f"{proba:.1%}")
            if proba >= 0.5:
                st.error("🔴 High Risk")
            elif proba >= 0.25:
                st.warning("🟡 Medium Risk")
            else:
                st.success("🟢 Low Risk")

        with col2:
            st.subheader("Why this prediction? (Local XAI)")
            shap_values = explainer(X_input)
            fig = plt.figure()
            shap.plots.waterfall(shap_values[0], max_display=12, show=False)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        with st.expander("See raw encoded features sent to the model"):
            st.dataframe(X_input.T.rename(columns={0: "value"}))
    else:
        st.info("Fill in the customer profile in the sidebar and click **Predict Churn Risk** to see results.")


# ---------------------------------------------------------------
# MODE 2: Batch CSV Upload (Global XAI)
# ---------------------------------------------------------------
elif app_mode == "Batch CSV Upload":
    st.subheader("Upload a CSV of customers")
    st.write("Please ensure the CSV has the raw columns: Age, Gender, Married, Dependents, Tenure in Months, Monthly Charge, etc.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
        
        st.write("Uploaded Data (First 5 Rows):")
        st.dataframe(input_df.head())
        
        # Rename columns (handling standard IBM Telco spaces) to match dictionary keys
        input_df_clean = input_df.rename(columns={
            'Tenure in Months': 'tenure', 'Monthly Charge': 'monthly_charge',
            'Total Charges': 'total_charges', 'Total Refunds': 'total_refunds',
            'Total Extra Data Charges': 'total_extra_data', 'Total Long Distance Charges': 'total_ld_charges',
            'Avg Monthly GB Download': 'avg_gb', 'Avg Monthly Long Distance Charges': 'avg_ld_charge',
            'Number of Referrals': 'num_referrals', 'Number of Dependents': 'num_dependents',
            'Age': 'age', 'Population': 'population', 'CLTV': 'cltv',
            'Gender': 'gender', 'Offer': 'offer', 'Internet Type': 'internet_type', 
            'Contract': 'contract', 'Payment Method': 'payment_method',
            'Married': 'married', 'Referred a Friend': 'referred', 
            'Phone Service': 'phone_service', 'Multiple Lines': 'multiple_lines',
            'Internet Service': 'internet_service', 'Online Security': 'online_security',
            'Online Backup': 'online_backup', 'Device Protection Plan': 'device_protection',
            'Premium Tech Support': 'premium_support', 'Streaming TV': 'streaming_tv',
            'Streaming Movies': 'streaming_movies', 'Streaming Music': 'streaming_music',
            'Unlimited Data': 'unlimited_data', 'Paperless Billing': 'paperless'
        })
        
        # Convert Yes/No to True/False for the checkboxes
        bool_map = {'Yes': True, 'No': False, 'True': True, 'False': False}
        cols_to_map = ['married', 'referred', 'phone_service', 'multiple_lines', 'internet_service', 
                       'online_security', 'online_backup', 'device_protection', 'premium_support', 
                       'streaming_tv', 'streaming_movies', 'streaming_music', 'unlimited_data', 'paperless']
        
        for col in cols_to_map:
            if col in input_df_clean.columns:
                input_df_clean[col] = input_df_clean[col].map(bool_map).fillna(False)

        # Safely drop rows with missing values in critical columns
        critical_cols = ['age', 'tenure', 'monthly_charge', 'total_charges']
        existing_critical = [c for c in critical_cols if c in input_df_clean.columns]
        if existing_critical:
            input_df_clean = input_df_clean.dropna(subset=existing_critical)

        batch_features = []
        errors = 0
        for _, row in input_df_clean.iterrows():
            row_dict = row.to_dict()
            try:
                X_input = build_feature_row(row_dict)
                batch_features.append(X_input)
            except Exception as e:
                errors += 1
                
        if errors > 0:
            st.warning(f"Skipped {errors} rows due to missing or incompatible data.")
            
        if batch_features:
            X_batch = pd.concat(batch_features, ignore_index=True)
            predictions = model.predict_proba(X_batch)[:, 1]
            input_df['Churn_Probability'] = predictions
            
            st.subheader("Batch Predictions")
            st.dataframe(input_df)
            
            csv = input_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Predictions", csv, "churn_predictions.csv", "text/csv")
            
            st.divider()
            
            st.subheader("Global XAI: SHAP Beeswarm Plot")
            st.write("This plot explains the model's behavior across *all* uploaded customers. Red dots mean a high feature value, Blue means low.")
            
            shap_values_batch = explainer(X_batch)
            
            fig_bee = plt.figure()
            shap.plots.beeswarm(shap_values_batch, max_display=15, show=False)
            st.pyplot(fig_bee, use_container_width=True)
            plt.close(fig_bee)
        else:
            st.error("No valid rows could be processed. Please check your CSV columns.")