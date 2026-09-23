
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

from lime.lime_tabular import LimeTabularExplainer
from pathlib import Path


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="CreditWise | Loan Decision Support",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# MODEL PATH
# ============================================================

CURRENT_DIR = Path(__file__).resolve().parent

if (CURRENT_DIR.parent / "models").exists():
    MODEL_DIR = CURRENT_DIR.parent / "models"
elif (CURRENT_DIR / "models").exists():
    MODEL_DIR = CURRENT_DIR / "models"
else:
    MODEL_DIR = Path("models")


# ============================================================
# CUSTOM CSS
# ============================================================
# HTML is NOT used for the visible application content.
# This CSS only styles native Streamlit components.
# ============================================================

st.markdown(
    """
    <style>

    /* ---------- PAGE ---------- */

    .stApp {
        background-color: #F5F7FB !important;
    }

    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1450px;
    }


    /* ---------- MAIN HEADINGS ---------- */

    h1, h2, h3, h4 {
        color: #172554 !important;
    }

    h1 {
        font-weight: 800 !important;
    }

    h2 {
        font-weight: 800 !important;
    }

    h3 {
        font-weight: 750 !important;
    }


    /* ---------- NORMAL TEXT ---------- */

    .main p,
    .main li {
        color: #334155 !important;
    }


    /* ---------- SIDEBAR ---------- */

    [data-testid="stSidebar"] {
        background-color: #EEF2FF !important;
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4 {
        color: #172554 !important;
    }

    [data-testid="stSidebar"] p {
        color: #334155 !important;
    }


    /* ---------- SIDEBAR INFO ---------- */

    [data-testid="stSidebar"] [data-testid="stAlert"] {
        background-color: #DBEAFE !important;
    }

    [data-testid="stSidebar"] [data-testid="stAlert"] p {
        color: #1E3A8A !important;
    }


    /* ---------- INPUT LABELS ---------- */

    [data-testid="stWidgetLabel"] p {
        color: #172554 !important;
        font-size: 14px !important;
        font-weight: 700 !important;
    }


    /* ---------- NUMBER INPUT ---------- */

    [data-testid="stNumberInput"] input {
        background-color: #FFFFFF !important;
        color: #172554 !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 10px !important;
        min-height: 42px;
        font-size: 15px !important;
    }

    [data-testid="stNumberInput"] button {
        background-color: #F8FAFC !important;
        color: #172554 !important;
        border-color: #CBD5E1 !important;
    }


    /* ---------- SELECT BOX ---------- */

    [data-testid="stSelectbox"] [data-baseweb="select"] > div {
        background-color: #FFFFFF !important;
        color: #172554 !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 10px !important;
        min-height: 42px;
    }

    [data-testid="stSelectbox"] span {
        color: #172554 !important;
    }

    [data-testid="stSelectbox"] svg {
        color: #172554 !important;
    }


    /* ---------- BUTTON ---------- */

    div.stButton > button {
        min-height: 52px;
        border-radius: 12px;
        font-size: 17px;
        font-weight: 700;
    }


    /* ---------- METRICS ---------- */

    [data-testid="stMetric"] {
        background-color: #FFFFFF !important;
        padding: 18px;
        border-radius: 14px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 3px 10px rgba(15, 23, 42, 0.05);
    }

    [data-testid="stMetricLabel"],
    [data-testid="stMetricLabel"] p {
        color: #64748B !important;
    }

    [data-testid="stMetricValue"] {
        color: #172554 !important;
    }


    /* ---------- TABS ---------- */

    .stTabs [data-baseweb="tab"] {
        color: #475569 !important;
        font-size: 16px;
        font-weight: 700;
    }

    .stTabs [aria-selected="true"] {
        color: #1D4ED8 !important;
    }


    /* ---------- DIVIDER ---------- */

    hr {
        border-color: #CBD5E1 !important;
    }


    /* ---------- ALERTS ---------- */

    [data-testid="stAlert"] p {
        color: #334155 !important;
    }


    /* ---------- FOOTER ---------- */

    .footer-note {
        text-align: center;
        color: #64748B !important;
        padding: 25px 0;
        margin-top: 30px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODEL AND PREPROCESSING
# ============================================================

@st.cache_resource
def load_models():

    model = joblib.load(
        MODEL_DIR / "final_catboost_model.pkl"
    )

    scaler = joblib.load(
        MODEL_DIR / "final_scaler.pkl"
    )

    ohe = joblib.load(
        MODEL_DIR / "final_ohe.pkl"
    )

    feature_columns = joblib.load(
        MODEL_DIR / "final_feature_columns.pkl"
    )

    X_train_scaled = joblib.load(
        MODEL_DIR / "final_X_train_scaled.pkl"
    )

    return (
        model,
        scaler,
        ohe,
        feature_columns,
        X_train_scaled
    )


try:

    (
        model,
        scaler,
        ohe,
        feature_columns,
        X_train_scaled
    ) = load_models()

except Exception as e:

    st.error("❌ Unable to load the CreditWise model files.")

    st.write("Make sure the following files exist inside the models folder:")

    st.code(
        """
final_catboost_model.pkl
final_scaler.pkl
final_ohe.pkl
final_feature_columns.pkl
final_X_train_scaled.pkl
        """
    )

    st.error(str(e))
    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("💳 CreditWise")

    st.caption(
        "Explainable AI Loan Decision Support"
    )

    st.divider()

    st.subheader("🧭 Dashboard")

    st.write("👤 Applicant Information")
    st.write("💰 Financial & Credit Details")
    st.write("📊 Loan Decision")
    st.write("📈 Risk Assessment")
    st.write("🧠 Explainable AI")
    st.write("💡 Recommendation")

    st.divider()

    st.subheader("⚙️ Model Information")

    st.info(
        """
**Algorithm:** CatBoost

**Test Accuracy:** 100%

**5-Fold CV Accuracy:** 99.9625%

**Explainability:** SHAP + LIME
        """
    )

    st.divider()

    st.caption("CreditWise Research Project")


# ============================================================
# MAIN HEADER
# ============================================================

st.title("💳 CreditWise")

st.subheader(
    "Explainable AI Framework for Loan Risk Assessment and Decision Support"
)

st.write(
    "🤖 Machine Learning  •  🔍 SHAP  •  🧠 LIME  •  📊 Risk Assessment"
)

st.divider()


# ============================================================
# APPLICANT INFORMATION
# ============================================================

st.header("👤 Applicant Information")

col1, col2, col3 = st.columns(3)


with col1:

    Age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=30,
        step=1
    )

    AnnualIncome = st.number_input(
        "Annual Income",
        min_value=0.0,
        value=50000.0,
        step=1000.0
    )

    CreditScore = st.number_input(
        "Credit Score",
        min_value=0.0,
        value=650.0,
        step=10.0
    )

    Experience = st.number_input(
        "Experience",
        min_value=0,
        value=5,
        step=1
    )

    LoanAmount = st.number_input(
        "Loan Amount",
        min_value=0.0,
        value=10000.0,
        step=1000.0
    )

    LoanDuration = st.number_input(
        "Loan Duration",
        min_value=0,
        value=36,
        step=1
    )


with col2:

    EmploymentStatus = st.selectbox(
        "Employment Status",
        ohe.categories_[0].tolist()
    )

    EducationLevel = st.selectbox(
        "Education Level",
        ohe.categories_[1].tolist()
    )

    LoanPurpose = st.selectbox(
        "Loan Purpose",
        ohe.categories_[2].tolist()
    )

    HomeOwnershipStatus = st.selectbox(
        "Home Ownership Status",
        ohe.categories_[3].tolist()
    )

    MaritalStatus = st.selectbox(
        "Marital Status",
        ohe.categories_[4].tolist()
    )

    NumberOfDependents = st.number_input(
        "Number of Dependents",
        min_value=0,
        value=0,
        step=1
    )


with col3:

    MonthlyDebtPayments = st.number_input(
        "Monthly Debt Payments",
        min_value=0.0,
        value=1000.0,
        step=100.0
    )

    CreditCardUtilizationRate = st.number_input(
        "Credit Card Utilization Rate",
        min_value=0.0,
        value=30.0,
        step=1.0
    )

    NumberOfOpenCreditLines = st.number_input(
        "Number of Open Credit Lines",
        min_value=0,
        value=5,
        step=1
    )

    NumberOfCreditInquiries = st.number_input(
        "Number of Credit Inquiries",
        min_value=0,
        value=2,
        step=1
    )

    DebtToIncomeRatio = st.number_input(
        "Debt To Income Ratio",
        min_value=0.0,
        value=0.30,
        step=0.01,
        format="%.2f"
    )

    BankruptcyHistory = st.selectbox(
        "Bankruptcy History",
        [0, 1]
    )


# ============================================================
# FINANCIAL AND CREDIT DETAILS
# ============================================================

st.header("💰 Financial & Credit Details")

col1, col2, col3 = st.columns(3)


with col1:

    PreviousLoanDefaults = st.number_input(
        "Previous Loan Defaults",
        min_value=0,
        value=0,
        step=1
    )

    PaymentHistory = st.number_input(
        "Payment History",
        min_value=0,
        value=50,
        step=1
    )

    LengthOfCreditHistory = st.number_input(
        "Length of Credit History",
        min_value=0,
        value=5,
        step=1
    )

    SavingsAccountBalance = st.number_input(
        "Savings Account Balance",
        min_value=0.0,
        value=10000.0,
        step=500.0
    )

    CheckingAccountBalance = st.number_input(
        "Checking Account Balance",
        min_value=0.0,
        value=5000.0,
        step=500.0
    )


with col2:

    TotalAssets = st.number_input(
        "Total Assets",
        min_value=0.0,
        value=50000.0,
        step=1000.0
    )

    TotalLiabilities = st.number_input(
        "Total Liabilities",
        min_value=0.0,
        value=20000.0,
        step=1000.0
    )

    MonthlyIncome = st.number_input(
        "Monthly Income",
        min_value=0.0,
        value=4000.0,
        step=100.0
    )

    UtilityBillsPaymentHistory = st.number_input(
        "Utility Bills Payment History",
        min_value=0.0,
        value=90.0,
        step=1.0
    )

    JobTenure = st.number_input(
        "Job Tenure",
        min_value=0,
        value=5,
        step=1
    )


with col3:

    NetWorth = st.number_input(
        "Net Worth",
        min_value=0.0,
        value=30000.0,
        step=1000.0
    )

    BaseInterestRate = st.number_input(
        "Base Interest Rate",
        min_value=0.0,
        value=5.0,
        step=0.1
    )

    InterestRate = st.number_input(
        "Interest Rate",
        min_value=0.0,
        value=7.0,
        step=0.1
    )

    MonthlyLoanPayment = st.number_input(
        "Monthly Loan Payment",
        min_value=0.0,
        value=500.0,
        step=50.0
    )

    TotalDebtToIncomeRatio = st.number_input(
        "Total Debt To Income Ratio",
        min_value=0.0,
        value=0.40,
        step=0.01,
        format="%.2f"
    )



# RiskScore is intentionally NOT collected from the user.
# It is excluded from the predictive feature set to prevent
# possible target leakage. Risk is calculated after prediction
# from the model's rejection probability.

# ============================================================
# PREDICT BUTTON
# ============================================================

st.write("")

predict_button = st.button(
    "🔍 Predict Loan Approval",
    type="primary",
    use_container_width=True
)


# ============================================================
# MAIN PREDICTION PIPELINE
# ============================================================

if predict_button:

    # ========================================================
    # RAW APPLICANT DATA
    # ========================================================

    applicant = pd.DataFrame(
        [{
            "Age": Age,
            "AnnualIncome": AnnualIncome,
            "CreditScore": CreditScore,
            "EmploymentStatus": EmploymentStatus,
            "EducationLevel": EducationLevel,
            "Experience": Experience,
            "LoanAmount": LoanAmount,
            "LoanDuration": LoanDuration,
            "MaritalStatus": MaritalStatus,
            "NumberOfDependents": NumberOfDependents,
            "HomeOwnershipStatus": HomeOwnershipStatus,
            "MonthlyDebtPayments": MonthlyDebtPayments,
            "CreditCardUtilizationRate": CreditCardUtilizationRate,
            "NumberOfOpenCreditLines": NumberOfOpenCreditLines,
            "NumberOfCreditInquiries": NumberOfCreditInquiries,
            "DebtToIncomeRatio": DebtToIncomeRatio,
            "BankruptcyHistory": BankruptcyHistory,
            "LoanPurpose": LoanPurpose,
            "PreviousLoanDefaults": PreviousLoanDefaults,
            "PaymentHistory": PaymentHistory,
            "LengthOfCreditHistory": LengthOfCreditHistory,
            "SavingsAccountBalance": SavingsAccountBalance,
            "CheckingAccountBalance": CheckingAccountBalance,
            "TotalAssets": TotalAssets,
            "TotalLiabilities": TotalLiabilities,
            "MonthlyIncome": MonthlyIncome,
            "UtilityBillsPaymentHistory": UtilityBillsPaymentHistory,
            "JobTenure": JobTenure,
            "NetWorth": NetWorth,
            "BaseInterestRate": BaseInterestRate,
            "InterestRate": InterestRate,
            "MonthlyLoanPayment": MonthlyLoanPayment,
            "TotalDebtToIncomeRatio": TotalDebtToIncomeRatio,
        }]
    )


    # ========================================================
    # FEATURE ENGINEERING
    # ========================================================

    applicant["DebtToIncomeRatio_sq"] = (
        applicant["DebtToIncomeRatio"] ** 2
    )

    applicant["CreditScore_sq"] = (
        applicant["CreditScore"] ** 2
    )


    # ========================================================
    # DROP ORIGINAL FEATURES
    # ========================================================

    applicant = applicant.drop(
        columns=[
            "CreditScore",
            "DebtToIncomeRatio"
        ]
    )


    # ========================================================
    # ONE-HOT ENCODING
    # ========================================================

    categorical_cols = [
        "EmploymentStatus",
        "EducationLevel",
        "LoanPurpose",
        "HomeOwnershipStatus",
        "MaritalStatus"
    ]

    encoded = ohe.transform(
        applicant[categorical_cols]
    )

    encoded_df = pd.DataFrame(
        encoded,
        columns=ohe.get_feature_names_out(categorical_cols),
        index=applicant.index
    )

    applicant = pd.concat(
        [
            applicant.drop(columns=categorical_cols),
            encoded_df
        ],
        axis=1
    )


    # ========================================================
    # MATCH TRAINING FEATURES
    # ========================================================

    applicant = applicant.reindex(
        columns=feature_columns,
        fill_value=0
    )


    # ========================================================
    # SCALE
    # ========================================================

    applicant_scaled = scaler.transform(
        applicant
    )


    # ========================================================
    # PREDICTION
    # ========================================================

    prediction = model.predict(
        applicant_scaled
    )[0]

    probability = model.predict_proba(
        applicant_scaled
    )[0]

    approval_percentage = probability[1] * 100
    rejection_percentage = probability[0] * 100
    rejection_probability = probability[0]


    # ========================================================
    # RISK ASSESSMENT
    # ========================================================

    if rejection_probability >= 0.50:
        risk_level = "High Risk"

    elif rejection_probability >= 0.20:
        risk_level = "Medium Risk"

    else:
        risk_level = "Low Risk"

    risk_score = rejection_probability * 100


    # ========================================================
    # LOAN DECISION
    # ========================================================

    st.divider()

    st.header("📊 Loan Decision")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Loan Status",
            "✅ APPROVED" if prediction == 1 else "❌ REJECTED"
        )

    with col2:

        st.metric(
            "Approval Probability",
            f"{approval_percentage:.2f}%"
        )

    with col3:

        st.metric(
            "Rejection Probability",
            f"{rejection_percentage:.2f}%"
        )


    if prediction == 1:

        st.success(
            """
### ✅ Loan Approved

The model predicts that the applicant is eligible for loan
approval based on the provided financial and credit information.
            """
        )

    else:

        st.error(
            """
### ❌ Loan Rejected

The model predicts that the applicant does not meet the
required approval criteria.
            """
        )


    # ========================================================
    # RISK ASSESSMENT
    # ========================================================

    st.header("📈 Applicant Risk Assessment")

    st.write(
        "Estimated rejection probability is used as the risk indicator."
    )

    st.progress(
        min(
            max(
                rejection_probability,
                0.0
            ),
            1.0
        )
    )

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Risk Score",
            f"{risk_score:.2f}/100"
        )

    with col2:

        st.metric(
            "Risk Level",
            risk_level
        )

    if risk_level == "High Risk":

        st.error(
            "🔴 High Risk — Significant risk indicators detected."
        )

    elif risk_level == "Medium Risk":

        st.warning(
            "🟠 Medium Risk — Additional financial review is recommended."
        )

    else:

        st.success(
            "🟢 Low Risk — The applicant shows relatively low estimated risk."
        )


    # ========================================================
    # APPLICANT PROFILE
    # ========================================================

    st.header("👤 Applicant Profile")

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Credit Score",
            f"{CreditScore:.0f}"
        )

    with col2:

        st.metric(
            "Annual Income",
            f"₹{AnnualIncome:,.0f}"
        )

    with col3:

        st.metric(
            "Loan Amount",
            f"₹{LoanAmount:,.0f}"
        )

    with col4:

        st.metric(
            "DTI Ratio",
            f"{DebtToIncomeRatio:.2f}"
        )


    # ========================================================
    # EXPLAINABLE AI
    # ========================================================

    st.header("🧠 Explainable AI")

    shap_tab, lime_tab = st.tabs(
        [
            "🔍 SHAP Explanation",
            "🧠 LIME Explanation"
        ]
    )


    # ========================================================
    # SHAP
    # ========================================================

    with shap_tab:

        st.subheader(
            "Why did the model make this decision?"
        )

        st.write(
            "SHAP explains how individual features contribute "
            "to the model prediction."
        )

        try:

            shap_explainer = shap.TreeExplainer(model)

            shap_result = shap_explainer.shap_values(
                applicant_scaled
            )

            shap_array = np.asarray(shap_result)

            if isinstance(shap_result, list):

                shap_values_local = np.asarray(
                    shap_result[1]
                )[0]

            elif shap_array.ndim == 3:

                shap_values_local = shap_array[
                    0,
                    :,
                    1
                ]

            elif shap_array.ndim == 2:

                shap_values_local = shap_array[0]

            else:

                shap_values_local = shap_array


            shap_df = pd.DataFrame(
                {
                    "Feature": feature_columns,
                    "SHAP Value": shap_values_local
                }
            )

            shap_df["Importance"] = (
                shap_df["SHAP Value"].abs()
            )

            shap_df = (
                shap_df
                .sort_values(
                    "Importance",
                    ascending=False
                )
                .head(10)
            )


            st.dataframe(
                shap_df[
                    [
                        "Feature",
                        "SHAP Value"
                    ]
                ],
                use_container_width=True,
                hide_index=True
            )


            st.subheader(
                "Top 10 Features Influencing Prediction"
            )

            fig, ax = plt.subplots(
                figsize=(9, 5)
            )

            ax.barh(
                shap_df["Feature"][::-1],
                shap_df["SHAP Value"][::-1]
            )

            ax.axvline(
                0,
                linewidth=1
            )

            ax.set_xlabel("SHAP Value")
            ax.set_ylabel("Feature")

            plt.tight_layout()

            st.pyplot(
                fig,
                use_container_width=True
            )

            plt.close(fig)


        except Exception as e:

            st.warning(
                "SHAP explanation could not be generated."
            )

            st.code(str(e))


    # ========================================================
    # LIME
    # ========================================================

    with lime_tab:

        st.subheader(
            "Local LIME Explanation"
        )

        st.write(
            "LIME provides a local approximation of the "
            "model decision for the selected applicant."
        )

        try:

            lime_explainer = LimeTabularExplainer(
                np.asarray(X_train_scaled),
                feature_names=feature_columns,
                class_names=[
                    "Rejected",
                    "Approved"
                ],
                mode="classification",
                random_state=42
            )

            lime_exp = lime_explainer.explain_instance(
                applicant_scaled[0],
                model.predict_proba,
                num_features=10
            )

            lime_data = lime_exp.as_list()

            lime_df = pd.DataFrame(
                lime_data,
                columns=[
                    "Feature",
                    "LIME Contribution"
                ]
            )

            st.dataframe(
                lime_df,
                use_container_width=True,
                hide_index=True
            )

            st.subheader(
                "LIME Feature Contributions"
            )

            lime_plot_df = lime_df.sort_values(
                "LIME Contribution"
            )

            fig, ax = plt.subplots(
                figsize=(9, 5)
            )

            ax.barh(
                lime_plot_df["Feature"],
                lime_plot_df["LIME Contribution"]
            )

            ax.axvline(
                0,
                linewidth=1
            )

            ax.set_xlabel("LIME Contribution")
            ax.set_ylabel("Feature")

            plt.tight_layout()

            st.pyplot(
                fig,
                use_container_width=True
            )

            plt.close(fig)


        except Exception as e:

            st.warning(
                "LIME explanation could not be generated."
            )

            st.code(str(e))


    # ========================================================
    # DECISION SUPPORT RECOMMENDATION
    # ========================================================

    st.header(
        "💡 Decision Support Recommendation"
    )


    if prediction == 1:

        if risk_level == "Low Risk":

            recommendation = (
                "The application is predicted to be approved "
                "with low estimated risk. The applicant's "
                "current financial profile supports the model "
                "decision."
            )

            action = "✅ RECOMMENDED FOR APPROVAL"


        elif risk_level == "Medium Risk":

            recommendation = (
                "The application is predicted to be approved, "
                "but additional financial review is recommended "
                "before final approval."
            )

            action = (
                "⚠️ APPROVAL WITH ADDITIONAL REVIEW"
            )


        else:

            recommendation = (
                "Although the model predicts approval, the "
                "applicant shows higher estimated risk "
                "indicators. Additional verification is "
                "recommended."
            )

            action = (
                "⚠️ ADDITIONAL VERIFICATION RECOMMENDED"
            )


    else:

        if risk_level == "High Risk":

            recommendation = (
                "The application is predicted to be rejected "
                "with high estimated risk. The applicant's "
                "financial profile requires careful review."
            )

            action = (
                "🔴 HIGH RISK — REVIEW REQUIRED"
            )


        else:

            recommendation = (
                "The application is predicted to be rejected. "
                "The major factors influencing the prediction "
                "should be reviewed before making the final "
                "decision."
            )

            action = (
                "⚠️ FURTHER REVIEW RECOMMENDED"
            )


    st.info(
        f"""
### {action}

{recommendation}
        """
    )


    # ========================================================
    # FINAL DECISION SUMMARY
    # ========================================================

    st.header(
        "📋 Final Decision Summary"
    )

    decision_text = (
        "Approved"
        if prediction == 1
        else "Rejected"
    )

    summary = pd.DataFrame(
        {
            "Decision": [
                decision_text
            ],

            "Approval Probability": [
                f"{approval_percentage:.2f}%"
            ],

            "Rejection Probability": [
                f"{rejection_percentage:.2f}%"
            ],

            "Risk Level": [
                risk_level
            ]
        }
    )

    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # EXPORT DECISION
    # ========================================================

    st.header(
        "📥 Export Decision"
    )

    report = pd.DataFrame(
        {
            "Parameter": [
                "Loan Decision",
                "Approval Probability",
                "Rejection Probability",
                "Risk Level",
                "Model",
                "Test Accuracy",
                "Cross-Validation Accuracy"
            ],

            "Value": [
                decision_text,
                f"{approval_percentage:.2f}%",
                f"{rejection_percentage:.2f}%",
                risk_level,
                "CatBoost",
                "100%",
                "99.9625%"
            ]
        }
    )

    csv_data = report.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="📥 Download Decision Report",
        data=csv_data,
        file_name="creditwise_decision_report.csv",
        mime="text/csv",
        use_container_width=True
    )


    # ========================================================
    # MODEL INFORMATION
    # ========================================================

    st.header(
        "⚙️ Model Information"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Model",
            "CatBoost"
        )

    with col2:

        st.metric(
            "Test Accuracy",
            "100%"
        )

    with col3:

        st.metric(
            "CV Accuracy",
            "99.9625%"
        )

    with col4:

        st.metric(
            "XAI",
            "SHAP + LIME"
        )


    # ========================================================
    # DISCLAIMER
    # ========================================================

    st.warning(
        """
⚠️ **Research Prototype Disclaimer**

This system provides a machine-learning-based decision-support
prediction. It should not be used as the sole basis for an actual
lending decision. Final decisions should follow institutional
policies and appropriate human review.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div class="footer-note">
        💳 <b>CreditWise</b><br>
        Explainable AI • Loan Risk Assessment • Decision Support<br><br>
        CatBoost • SHAP • LIME • Streamlit<br><br>
        Research Prototype — Machine Learning Based Loan Decision Support
    </div>
    """,
    unsafe_allow_html=True
)
