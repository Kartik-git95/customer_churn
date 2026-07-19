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
import plotly.graph_objects as go
import shap
import joblib

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📡", layout="wide")

# ---------------------------------------------------------------
# Theme: IBM Plex type system + a "signal room" navy/coral/teal
# palette (a nod to the IBM Telco dataset itself, rather than a
# generic dark-mode default).
# ---------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root {
    --bg: #0D1321;
    --surface: #161D30;
    --surface-2: #232C48;
    --text: #E8ECF4;
    --text-muted: #8B93A7;
    --accent-primary: #6C8EF5;
    --accent-safe: #2DD4BF;
    --accent-warn: #F5A623;
    --accent-risk: #F2545B;
}

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
.stApp { background: var(--bg); }
h1, h2, h3, h4, h5 { font-family: 'IBM Plex Sans', sans-serif; font-weight: 700 !important; color: var(--text) !important; }
p, span, label, div { color: var(--text); }

/* Hero */
.hero { display: flex; align-items: center; gap: 16px; margin-bottom: 2px; }
.hero-icon { font-size: 40px; line-height: 1; }
.hero-title { font-size: 32px; font-weight: 700; color: var(--text); letter-spacing: -0.5px; }
.hero-subtitle { font-size: 13px; color: var(--text-muted); font-family: 'IBM Plex Mono', monospace; margin-top: 3px; }
.hero-divider { height: 3px; width: 100%; margin: 16px 0 26px 0; border-radius: 3px;
    background: linear-gradient(90deg, var(--accent-primary), var(--accent-safe)); }

/* Bordered containers -> cards */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--surface) !important;
    border: 1px solid var(--surface-2) !important;
    border-radius: 14px !important;
}
[data-testid="stForm"] {
    background: var(--surface);
    border: 1px solid var(--surface-2);
    border-radius: 14px;
    padding: 6px 14px 14px 14px;
}

/* Custom stat / insight cards */
.stat-card { background: var(--surface); border: 1px solid var(--surface-2); border-radius: 14px;
    padding: 18px 20px; }
.stat-label { font-size: 11px; text-transform: uppercase; letter-spacing: 1.2px; color: var(--text-muted);
    font-family: 'IBM Plex Mono', monospace; margin-bottom: 8px; }
.stat-value { font-size: 28px; font-weight: 600; font-family: 'IBM Plex Mono', monospace; }

.insight-box { background: var(--surface-2); border-left: 3px solid var(--accent-primary);
    padding: 14px 18px; border-radius: 8px; font-size: 15px; line-height: 1.55; color: var(--text); }

.risk-badge { display: inline-flex; align-items: center; gap: 8px; padding: 7px 16px; border-radius: 999px;
    font-family: 'IBM Plex Mono', monospace; font-weight: 600; font-size: 14px; margin-top: 8px; }

/* Native widgets */
[data-testid="stMetricValue"] { font-family: 'IBM Plex Mono', monospace; color: var(--accent-primary); }
.stButton button, .stDownloadButton button, [data-testid="stFormSubmitButton"] button {
    background: linear-gradient(90deg, var(--accent-primary), #8AA5F8);
    color: #0D1321; font-weight: 600; border: none; border-radius: 10px; padding: 10px 18px;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.stButton button:hover, .stDownloadButton button:hover, [data-testid="stFormSubmitButton"] button:hover {
    transform: translateY(-1px); box-shadow: 0 6px 18px rgba(108, 142, 245, 0.35); color: #0D1321;
}
.stTabs [data-baseweb="tab"] { font-family: 'IBM Plex Sans', sans-serif; font-weight: 600; font-size: 15px; }
.stTabs [aria-selected="true"] { color: var(--accent-primary) !important; }

.footer { margin-top: 44px; padding-top: 16px; border-top: 1px solid var(--surface-2);
    font-size: 12px; color: var(--text-muted); font-family: 'IBM Plex Mono', monospace; text-align: center; }
</style>
""", unsafe_allow_html=True)


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
# Plain-English explanation from SHAP values
# ---------------------------------------------------------------
def describe_feature(name: str, value, shap_val: float) -> str:
    if name == "Tenure_in_Months":
        if value <= 3:
            return f"being a brand-new customer (only {int(value)} month(s) in)"
        elif value < 24:
            return f"having a fairly short tenure ({int(value)} months)"
        return f"having {value / 12:.1f} years of tenure"
    if name == "Contract_Two Year":
        return "being locked into a two-year contract" if value else "not having a long-term (two-year) contract"
    if name == "Contract_One Year":
        return "being on a one-year contract" if value else "not having a one-year commitment"
    if name.startswith("Internet_Type_"):
        net_type = name.replace("Internet_Type_", "")
        return f"being on the {net_type} internet plan" if value else f"not being on {net_type} internet"
    if name == "Online_Security":
        return "having online security enabled" if value else "not having the online security add-on"
    if name == "Device_Protection_Plan":
        return "having a device protection plan" if value else "not having device protection"
    if name == "Premium_Tech_Support":
        return "having premium tech support" if value else "not having premium tech support"
    if name == "Online_Backup":
        return "having online backup enabled" if value else "not having online backup"
    if name == "Monthly_Charge":
        return f"paying ${value:.2f} a month"
    if name == "Age":
        return f"being {int(value)} years old"
    if name == "Number_of_Referrals":
        return "never having referred anyone" if value == 0 else f"having referred {int(value)} friend(s)"
    if name == "Referred_a_Friend":
        return "having referred a friend at least once" if value else "never having referred a friend"
    if name == "Paperless_Billing":
        return "using paperless billing" if value else "not using paperless billing"
    if name == "TotalServicesSubscribed":
        return (f"subscribing to only {int(value)} add-on service(s)" if value <= 1
                else f"subscribing to {int(value)} add-on services")
    if name.startswith("Payment_Method_"):
        method = name.replace("Payment_Method_", "")
        return f"paying via {method}" if value else f"not paying via {method}"
    if name == "CLTV":
        return f"having a customer lifetime value score of {int(value)}"
    clean_name = name.replace("_", " ")
    return f"their {clean_name} ({value})"


def _join_list(items: list) -> str:
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + ", and " + items[-1]


def build_explanation_sentence(shap_row_values, feature_names, feature_values, proba, top_n=4) -> str:
    contributions = list(zip(feature_names, feature_values, shap_row_values))
    contributions.sort(key=lambda x: abs(x[2]), reverse=True)
    top = contributions[:top_n]

    risk_up = [describe_feature(n, v, s) for n, v, s in top if s > 0]
    risk_down = [describe_feature(n, v, s) for n, v, s in top if s < 0]

    sentence = f"This customer has a <strong>{proba:.0%}</strong> predicted chance of churning."
    if risk_up:
        sentence += f" The model's biggest concerns are {_join_list(risk_up)}."
    if risk_down:
        verb = "is" if len(risk_down) == 1 else "are"
        sentence += f" On the positive side, {_join_list(risk_down)} {verb} helping keep them retained."
    return sentence


# ---------------------------------------------------------------
# Visual helpers
# ---------------------------------------------------------------
def risk_tier(proba):
    if proba >= 0.5:
        return "#F2545B", "rgba(242,84,91,0.15)", "High Risk", "🔴"
    elif proba >= 0.25:
        return "#F5A623", "rgba(245,166,35,0.15)", "Medium Risk", "🟡"
    return "#2DD4BF", "rgba(45,212,191,0.15)", "Low Risk", "🟢"


def risk_badge_html(proba):
    color, bg, label, icon = risk_tier(proba)
    return f'<div class="risk-badge" style="color:{color}; background:{bg};">{icon} {label}</div>'


def render_gauge(proba):
    bar_color, _, _, _ = risk_tier(proba)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=proba * 100,
        number={'suffix': "%", 'font': {'size': 40, 'family': "IBM Plex Mono", 'color': '#E8ECF4'}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': '#8B93A7', 'tickfont': {'color': '#8B93A7', 'size': 10}},
            'bar': {'color': bar_color, 'thickness': 0.32},
            'bgcolor': "#161D30",
            'borderwidth': 0,
            'steps': [
                {'range': [0, 25], 'color': 'rgba(45,212,191,0.15)'},
                {'range': [25, 50], 'color': 'rgba(245,166,35,0.15)'},
                {'range': [50, 100], 'color': 'rgba(242,84,91,0.15)'},
            ],
        }
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=200, margin=dict(l=25, r=25, t=25, b=10), font={'color': '#E8ECF4'},
    )
    return fig


def stat_card_html(label, value, color="#6C8EF5"):
    return f"""<div class="stat-card">
        <div class="stat-label">{label}</div>
        <div class="stat-value" style="color:{color}">{value}</div>
    </div>"""


# ---------------------------------------------------------------
# Hero
# ---------------------------------------------------------------
st.markdown("""
<div class="hero">
    <div class="hero-icon">📡</div>
    <div>
        <div class="hero-title">Customer Churn Predictor</div>
        <div class="hero-subtitle">XGBoost + SHAP · IBM Telco Customer Churn dataset</div>
    </div>
</div>
<div class="hero-divider"></div>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["👤  Single Customer", "📁  Batch CSV Upload"])


# ---------------------------------------------------------------
# TAB 1: Single Customer Form (Local XAI)
# ---------------------------------------------------------------
with tab1:
    with st.sidebar.form("customer_form"):
        st.markdown("#### Customer Profile")

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

        submitted = st.form_submit_button("⚡ Predict Churn Risk", use_container_width=True)

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
        shap_values = explainer(X_input)

        with st.container(border=True):
            gauge_col, insight_col = st.columns([1, 1.3])

            with gauge_col:
                st.plotly_chart(render_gauge(proba), use_container_width=True, config={'displayModeBar': False})
                st.markdown(risk_badge_html(proba), unsafe_allow_html=True)

            with insight_col:
                st.markdown("#### Why this prediction?")
                sentence = build_explanation_sentence(
                    shap_values[0].values, FEATURE_NAMES, X_input.iloc[0].values, proba, top_n=4
                )
                st.markdown(f'<div class="insight-box">{sentence}</div>', unsafe_allow_html=True)

            st.write("")
            with st.expander("🔍 Technical SHAP waterfall"):
                fig = plt.figure()
                shap.plots.waterfall(shap_values[0], max_display=12, show=False)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

            with st.expander("📋 Raw encoded features sent to the model"):
                st.dataframe(X_input.T.rename(columns={0: "value"}), use_container_width=True)
    else:
        st.info("Fill in the customer profile in the sidebar and click **⚡ Predict Churn Risk** to see results.")


# ---------------------------------------------------------------
# TAB 2: Batch CSV Upload (Global XAI)
# ---------------------------------------------------------------
with tab2:
    st.markdown("#### Upload a CSV of customers")
    st.caption("Expects the raw IBM Telco-style columns: Age, Gender, Married, Dependents, "
               "Tenure in Months, Monthly Charge, etc.")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv", label_visibility="collapsed")

    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)

        with st.expander("Preview uploaded data (first 5 rows)", expanded=False):
            st.dataframe(input_df.head(), use_container_width=True)

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

        bool_map = {'Yes': True, 'No': False, 'True': True, 'False': False}
        cols_to_map = ['married', 'referred', 'phone_service', 'multiple_lines', 'internet_service',
                       'online_security', 'online_backup', 'device_protection', 'premium_support',
                       'streaming_tv', 'streaming_movies', 'streaming_music', 'unlimited_data', 'paperless']

        for col in cols_to_map:
            if col in input_df_clean.columns:
                input_df_clean[col] = input_df_clean[col].map(bool_map).fillna(False)

        critical_cols = ['age', 'tenure', 'monthly_charge', 'total_charges']
        existing_critical = [c for c in critical_cols if c in input_df_clean.columns]
        if existing_critical:
            input_df_clean = input_df_clean.dropna(subset=existing_critical)

        batch_features = []
        errors = 0
        for _, row in input_df_clean.iterrows():
            row_dict = row.to_dict()
            try:
                X_row = build_feature_row(row_dict)
                batch_features.append(X_row)
            except Exception:
                errors += 1

        if errors > 0:
            st.warning(f"Skipped {errors} rows due to missing or incompatible data.")

        if batch_features:
            X_batch = pd.concat(batch_features, ignore_index=True)
            predictions = model.predict_proba(X_batch)[:, 1]
            input_df['Churn_Probability'] = predictions

            total = len(predictions)
            churners = int((predictions >= 0.5).sum())
            avg_risk = predictions.mean()

            st.write("")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(stat_card_html("Customers Analyzed", f"{total:,}", "#6C8EF5"), unsafe_allow_html=True)
            with c2:
                st.markdown(stat_card_html("Predicted Churners", f"{churners:,} ({churners / total:.0%})", "#F2545B"), unsafe_allow_html=True)
            with c3:
                st.markdown(stat_card_html("Average Risk", f"{avg_risk:.1%}", "#F5A623"), unsafe_allow_html=True)

            st.write("")
            st.markdown("#### Batch Predictions")
            st.dataframe(input_df, use_container_width=True)

            csv = input_df.to_csv(index=False).encode('utf-8')
            st.download_button("⬇ Download Predictions", csv, "churn_predictions.csv", "text/csv")

            st.divider()

            st.markdown("#### Global XAI: SHAP Beeswarm Plot")
            st.caption("Explains the model's behavior across *all* uploaded customers. "
                       "Red = high feature value, blue = low feature value.")

            shap_values_batch = explainer(X_batch)
            fig_bee = plt.figure()
            shap.plots.beeswarm(shap_values_batch, max_display=15, show=False)
            st.pyplot(fig_bee, use_container_width=True)
            plt.close(fig_bee)
        else:
            st.error("No valid rows could be processed. Please check your CSV columns.")

st.markdown('<div class="footer">Built with XGBoost · SHAP · Streamlit — IBM Telco Customer Churn dataset</div>',
            unsafe_allow_html=True)