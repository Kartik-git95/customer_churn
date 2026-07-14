# Customer Churn Prediction — Telco Dataset (IBM/Kaggle Extended Version)

Predicting which customers are likely to churn, using a leakage-aware feature set, XGBoost, and SHAP-based explainability.

---

## 📌 Overview

Customer churn is one of the most expensive problems in subscription-style businesses — it's far cheaper to retain an existing customer than acquire a new one. This project builds a churn prediction model on the IBM Telco Customer Churn dataset (the richer, extended Kaggle version with 50 raw columns rather than the basic 21-column version), with a strong emphasis on:

- **Data leakage prevention** — several columns in this dataset directly encode the answer, and identifying them was a core part of the work
- **Model explainability** — using SHAP to explain individual predictions and validate them against dataset-wide patterns, not just report a single accuracy number

## 🎯 Problem Statement

[1-2 sentences on the business framing — e.g., "Telecom companies lose significant revenue to customer churn. This project identifies at-risk customers early enough for a retention team to intervene, and explains *why* each customer is flagged so interventions can be targeted."]

## 📊 Dataset

- **Source:** IBM Telco Customer Churn (extended Kaggle version) — [INSERT KAGGLE LINK]
- **Size:** 7,043 customers, 50 raw columns
- **Target:** `Churn_Label` (Yes/No)
- **Class balance:** [INSERT % churned vs retained from your Step 5 value_counts output]

## 🧹 Data Cleaning & Leakage Prevention

This dataset has several traps that aren't present in the basic Telco CSV most tutorials use. Identifying and removing them was a deliberate step, not an afterthought:

| Column(s) | Reason for removal |
|---|---|
| `Churn Score` | IBM's own pre-computed churn propensity score — effectively a proxy for the target |
| `Customer Status` | Directly encodes Churned/Stayed/Joined — same information as the target |
| `Churn Category`, `Churn Reason` | Only populated for the ~1,869 customers who already churned — these describe an outcome that hasn't happened yet at prediction time |
| `Customer ID` | Pure identifier, no predictive value |
| `Country`, `State`, `Quarter` | Constant across all 7,043 rows (confirmed via `nunique()`) — zero information |
| `City`, `Zip Code` | High-cardinality (1,106 / 1,626 unique values) — too granular to generalize from, high overfitting risk |
| `Latitude`, `Longitude` | Dropped as noise for tree-based models; not used for geo-clustering in this version |

**Borderline cases, evaluated rather than assumed:**
- `CLTV` and `Satisfaction_Score` — kept, but flagged and tested for leakage risk. [INSERT: state whether you re-ran the model without them and whether AUC changed]
- `Total_Revenue` — kept, but noted as highly collinear with `Total_Charges` + `Total_Long_Distance_Charges` − `Total_Refunds`, which dilutes SHAP attribution across correlated features rather than concentrating it.

## 🕳️ Missing Data Handling

Nulls in this dataset weren't random — they meant something:

- `Offer` (3,877 nulls) → filled with `"No Offer"`. These weren't missing data, they were customers who were never extended a promotional offer.
- `Internet_Type` (1,526 nulls) → verified every null row had `Internet_Service == "No"`, then filled with `"No Internet Service"`. Confirmed the assumption before filling rather than guessing.

## 🛠️ Feature Engineering

| Feature | Description |
|---|---|
| `AvgChargesPerMonth` | `Total_Charges / Tenure_in_Months` — sanity-checked against `Monthly_Charge` |
| `TenureGroup` | Binned tenure (0–1yr, 1–2yr, 2–4yr, 4+yr) to help tree models capture non-linear tenure effects |
| `HasSecurityBundle` | Combined flag for customers with both online security and premium tech support |
| `TotalServicesSubscribed` | Count of add-on services subscribed (security, backup, protection, support, streaming ×3) |
| `HasRefund` | Binary flag for customers with `Total_Refunds > 0` — a feature only available in this extended dataset version |

## 🤖 Model

- **Algorithm:** XGBoost Classifier
- **Encoding:** One-hot encoding of categorical features (`pd.get_dummies`)
- **Class imbalance handling:** [INSERT — SMOTE / `scale_pos_weight` / which was actually used]
- **Train/test split:** [INSERT split ratio and whether stratified]

## 📈 Model Performance

[INSERT YOUR METRICS TABLE HERE — accuracy, precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix. This is the one section I don't have numbers for from our conversation — paste your Day 2 output here.]

## 🔍 Model Explainability (SHAP)

Rather than stopping at aggregate metrics, this project uses SHAP to explain both global feature importance and individual predictions — and checks that individual explanations actually agree with dataset-wide patterns.

**Global drivers (beeswarm summary):**
- `Number_of_Referrals` is the single strongest predictor by a wide margin — high referral counts are strongly protective against churn
- Contract length (`Contract_Two_Year`, `Contract_One_Year`) shows clean, expected separation — longer contracts reduce churn risk
- `Tenure_in_Months` — low tenure clearly increases risk, consistent with new customers churning most
- `Age` — older customers show consistently higher churn risk across the full test set, not just isolated cases

**Two contrasting individual examples:**

*High-risk customer (predicted 98.8% churn probability, actually churned):* A brand-new customer (1 month tenure), month-to-month contract, fiber internet, no security/support add-ons, and — notably — age 71. Every major SHAP contributor for this customer pushed risk upward; there was no protective factor in their profile at all.

*Low-risk customer (predicted 0.13% churn probability, actually retained):* Two-year contract, 52 months tenure, and 10 referrals — by far the largest single protective factor SHAP identified across the whole analysis.

**Notable finding — correlated-feature disagreement:** `Referred_a_Friend` (binary) and `Number_of_Referrals` (count) point in *opposite* directions in the global SHAP plot, despite describing the same underlying behavior. This is a known SHAP symptom of correlated features splitting attribution unevenly, and is flagged below as a future improvement (VIF check / feature consolidation) rather than treated as a real contradictory insight.

## 🖥️ Deployment (Streamlit App)

[INSERT — link to live app once deployed, or "In progress" if Day 5 isn't complete yet. Briefly describe what the app lets a user do — e.g., input a customer profile and get a churn probability + SHAP explanation.]

## 💡 Key Insights & Business Recommendations

- **Contract incentives matter most.** Since month-to-month status is one of the strongest churn drivers, promotional incentives for 1-2 year contract upgrades could meaningfully reduce churn.
- **Referral programs double as retention signals.** Customers with multiple referrals are dramatically less likely to churn — worth investigating whether referral programs should be actively promoted to at-risk segments.
- **New customers need an onboarding safety net.** Tenure under 12 months is consistently high-risk; a structured onboarding/check-in process in the first year could catch churn before it happens.

## 🧰 Tech Stack

Python · pandas · scikit-learn · XGBoost · SHAP · Streamlit · matplotlib / seaborn

## ⚙️ How to Run

```bash
# INSERT once repo structure is finalized, e.g.:
git clone [your-repo-url]
cd [repo-name]
pip install -r requirements.txt
streamlit run app.py
```

## 🔮 Future Improvements

- Run a VIF (variance inflation factor) check between `Referred_a_Friend` and `Number_of_Referrals`; likely drop one to clean up attribution
- Quantify the AUC impact of excluding `CLTV` and `Satisfaction_Score` rather than just flagging them as borderline
- Consider dropping or consolidating `Total_Revenue` given its collinearity with existing charge-related features
- Expand the Streamlit app to accept live SHAP explanations per user-submitted profile

## 📁 Project Structure

```
[INSERT once finalized — e.g.:]
├── data/
│   └── cleaned_churn_data.csv
├── notebooks/
│   └── churn_analysis.ipynb
├── models/
│   ├── churn_model.pkl
│   └── model_columns.pkl
├── app.py
├── requirements.txt
└── README.md
```

## 👤 Author

[kartik padmanabhan] · [LinkedIn] · [Kartik-git9b] · 
