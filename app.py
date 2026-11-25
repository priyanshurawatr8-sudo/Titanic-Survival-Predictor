import streamlit as st
import joblib
import pandas as pd

st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢")

@st.cache_resource
def load_model():
    data = joblib.load("titanic_model.joblib")
    return data["model"], data["feature_names"]

model, feature_names = load_model()

st.title("🚢 Titanic Survival Predictor")
st.write("Enter passenger details to predict survival.")

pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", 1, 80, 25)
sibsp = st.slider("Siblings/Spouses", 0, 8, 0)
parch = st.slider("Parents/Children", 0, 6, 0)
fare = st.number_input("Fare", min_value=0.0, value=32.0)
embarked = st.selectbox("Port of Embarkation", ["Cherbourg", "Queenstown", "Southampton"])

if st.button("Predict Survival"):
    passenger = {
        "pclass": pclass,
        "age": age,
        "sibsp": sibsp,
        "parch": parch,
        "fare": fare,
        "sex_male": 1 if sex == "Male" else 0,
        "embarked_Q": 1 if embarked == "Queenstown" else 0,
        "embarked_S": 1 if embarked == "Southampton" else 0
    }

    input_data = pd.DataFrame([passenger])[feature_names]
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.success("Prediction: Passenger may survive! 🎉")
    else:
        st.error("Prediction: Passenger may not survive.")

    st.caption("This is a machine learning prediction, not a certainty.")
