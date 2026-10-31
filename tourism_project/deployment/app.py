"""Streamlit app - Wellness Tourism Package purchase prediction.

Loads the model that the GitHub Actions pipeline trained and committed next to
this file, collects the customer's details, puts them into a DataFrame (raw
values - the preprocessing lives inside the saved model pipeline) and shows the
predicted purchase probability.
"""
import os

import joblib
import pandas as pd
import streamlit as st

# Model file written by tourism_project/model_building/train.py
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "best_tourism_package_model_v1.joblib"
)

# Decision threshold - keep identical to classification_threshold in train.py
CLASSIFICATION_THRESHOLD = 0.45

# In the training data every pitched product is always paired with one designation
PRODUCT_TO_DESIGNATION = {
    "Basic": "Executive",
    "Standard": "Senior Manager",
    "Deluxe": "Manager",
    "Super Deluxe": "AVP",
    "King": "VP",
}

st.set_page_config(page_title="Tourism Package Prediction", page_icon="🧳")


@st.cache_resource
def load_model(path):
    """Load the trained scikit-learn pipeline (preprocessing + XGBoost) once."""
    return joblib.load(path)


def format_probability(p):
    """Readable percentage (avoids showing a literal 0.0% / 100.0%)."""
    if p < 0.001:
        return "<0.1%"
    if p > 0.999:
        return ">99.9%"
    return f"{p:.1%}"


st.title("Tourism Package Prediction")
st.write(
    "Fill in the customer details below to predict whether the customer will "
    "purchase the Wellness Tourism Package."
)

if not os.path.exists(MODEL_PATH):
    st.error(
        "Model file not found. Run the 'Tourism Project Pipeline' workflow in the "
        "GitHub Actions tab - it trains the model and commits it to this "
        "repository - then reload this page."
    )
    st.stop()

model = load_model(MODEL_PATH)

# ----------------------------------------------------------------------------
# Collect user input (value ranges follow the training data)
# ----------------------------------------------------------------------------
st.subheader("Customer details")
col1, col2 = st.columns(2)

with col1:
    Age = st.slider("Age", 18, 65, 36)
    Gender = st.selectbox("Gender", ["Male", "Female"])
    MaritalStatus = st.selectbox("Marital Status", ["Married", "Single", "Divorced", "Unmarried"])
    Occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
    Designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
    MonthlyIncome = st.number_input("Monthly Income", min_value=1000.0, max_value=200000.0, value=22000.0, step=500.0)
    CityTier = st.selectbox("City Tier", [1, 2, 3])
    Passport = st.selectbox("Has Passport?", ["No", "Yes"])
    OwnCar = st.selectbox("Owns a Car?", ["Yes", "No"])

with col2:
    NumberOfPersonVisiting = st.slider("Number of Persons Visiting", 1, 5, 3)
    NumberOfChildrenVisiting = st.slider("Number of Children Visiting (under 5)", 0, 3, 1)
    NumberOfTrips = st.slider("Average Trips per Year", 1, 22, 3)
    PreferredPropertyStar = st.selectbox("Preferred Property Star", [3, 4, 5])
    TypeofContact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
    ProductPitched = st.selectbox("Product Pitched", list(PRODUCT_TO_DESIGNATION))
    DurationOfPitch = st.slider("Duration of Pitch (minutes)", 5, 130, 14)
    NumberOfFollowups = st.slider("Number of Follow-ups", 1, 6, 4)
    PitchSatisfactionScore = st.slider("Pitch Satisfaction Score", 1, 5, 3)

if PRODUCT_TO_DESIGNATION[ProductPitched] != Designation:
    st.warning(
        f"In the training data '{ProductPitched}' is only ever pitched to "
        f"'{PRODUCT_TO_DESIGNATION[ProductPitched]}' customers, so this combination "
        "is outside what the model has seen and the result is less reliable."
    )

# ----------------------------------------------------------------------------
# Put the inputs into a DataFrame (column names must match the training data)
# ----------------------------------------------------------------------------
input_data = pd.DataFrame([{
    "Age": Age,
    "TypeofContact": TypeofContact,
    "CityTier": CityTier,
    "DurationOfPitch": DurationOfPitch,
    "Occupation": Occupation,
    "Gender": Gender,
    "NumberOfPersonVisiting": NumberOfPersonVisiting,
    "NumberOfFollowups": NumberOfFollowups,
    "ProductPitched": ProductPitched,
    "PreferredPropertyStar": PreferredPropertyStar,
    "MaritalStatus": MaritalStatus,
    "NumberOfTrips": NumberOfTrips,
    "Passport": 1 if Passport == "Yes" else 0,
    "PitchSatisfactionScore": PitchSatisfactionScore,
    "OwnCar": 1 if OwnCar == "Yes" else 0,
    "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
    "Designation": Designation,
    "MonthlyIncome": MonthlyIncome,
}])

with st.expander("Input sent to the model (DataFrame)"):
    st.dataframe(input_data)

# ----------------------------------------------------------------------------
# Predict
# ----------------------------------------------------------------------------
if st.button("Predict"):
    prob = float(model.predict_proba(input_data)[0, 1])
    prediction = int(prob >= CLASSIFICATION_THRESHOLD)

    st.metric("Purchase probability", format_probability(prob))
    if prediction == 1:
        st.success("Prediction: the customer is LIKELY to purchase the Wellness Tourism Package.")
    else:
        st.info("Prediction: the customer is UNLIKELY to purchase the Wellness Tourism Package.")
    st.caption(f"Decision threshold: {CLASSIFICATION_THRESHOLD:.0%} predicted purchase probability.")
