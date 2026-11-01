"""Streamlit demo for pre-contact tourism package purchase scoring."""
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).parent / "best_tourism_package_model_v1.joblib"
THRESHOLD = 0.50

st.set_page_config(page_title="Tourism Package Lead Scoring", page_icon="🧳", layout="wide")
st.title("Tourism Package Lead Scoring")
st.caption("Profile-only estimate intended to prioritize outreach before a sales pitch.")
st.info(
    "This academic demo excludes contact and pitch outcomes. A probability is not a guarantee; "
    "review fairness and local privacy rules before operational use."
)

if not MODEL_PATH.exists():
    st.error("Trained model not found. Run the Tourism Project Pipeline in GitHub Actions first.")
    st.stop()

@st.cache_resource
def load_model(path):
    return joblib.load(path)

model = load_model(str(MODEL_PATH))

with st.sidebar.form("customer_profile"):
    st.header("Customer profile")
    age = st.slider("Age", 18, 61, 36)
    gender = st.selectbox("Gender", ["Female", "Male"])
    marital_status = st.selectbox("Marital status", ["Single", "Married", "Divorced", "Unmarried"])
    occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
    designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
    monthly_income = st.number_input("Monthly income", min_value=1000.0, max_value=98678.0, value=22000.0, step=500.0)
    city_tier = st.selectbox("City tier", [1, 2, 3])
    number_of_person_visiting = st.slider("People in planned trip", 1, 5, 3)
    number_of_children_visiting = st.slider("Children under 5 in planned trip", 0, 3, 0)
    number_of_trips = st.slider("Trips per year", 1, 22, 3)
    preferred_property_star = st.selectbox("Preferred property stars", [3, 4, 5])
    passport = st.selectbox("Has passport?", ["No", "Yes"])
    own_car = st.selectbox("Owns a car?", ["No", "Yes"])
    submit = st.form_submit_button("Estimate purchase likelihood", type="primary")

if submit:
    input_data = pd.DataFrame([{
        "Age": age,
        "CityTier": city_tier,
        "NumberOfPersonVisiting": number_of_person_visiting,
        "PreferredPropertyStar": preferred_property_star,
        "NumberOfTrips": number_of_trips,
        "Passport": 1 if passport == "Yes" else 0,
        "OwnCar": 1 if own_car == "Yes" else 0,
        "NumberOfChildrenVisiting": number_of_children_visiting,
        "MonthlyIncome": monthly_income,
        "Occupation": occupation,
        "Gender": gender,
        "MaritalStatus": marital_status,
        "Designation": designation,
    }])
    probability = float(model.predict_proba(input_data)[0, 1])
    prediction = int(probability >= THRESHOLD)
    left, right = st.columns(2)
    left.metric("Estimated purchase probability", f"{probability:.1%}")
    right.metric("At 0.50 decision threshold", "Prioritize for outreach" if prediction else "Lower predicted likelihood")
    with st.expander("Input sent to the model"):
        st.dataframe(input_data, use_container_width=True)
