import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Ρύθμιση Σελίδας
st.set_page_config(page_title="Weather Predictor", page_icon="🌤️")

st.title("🌤️ Πρόβλεψη Βροχής για Αύριο")
st.markdown("Συμπληρώστε τα παρακάτω στοιχεία για να μάθετε την πιθανότητα βροχής.")


# Φόρτωση Μοντέλων
@st.cache_resource
def load_assets():
    model = joblib.load("../models/best_model.pkl")
    scaler = joblib.load("../models/scaler.pkl")
    features = joblib.load("../models/feature_names.pkl")
    return model, scaler, features


model, scaler, feature_names = load_assets()

# 3. Φόρμα Εισαγωγής
st.subheader("📍 Στοιχεία Καιρού")

col1, col2 = st.columns(2)

with col1:
    month = st.selectbox("Μήνας", range(1, 13), index=5)
    min_temp = st.number_input("Ελάχιστη Θερμοκρασία (°C)", value=15.0)
    max_temp = st.number_input("Μέγιστη Θερμοκρασία (°C)", value=25.0)

    rain_today = st.selectbox("Έβρεξε σήμερα;", ["Όχι", "Ναι"])

    # Υπό συνθήκες εμφάνιση όγκου βροχής
    if rain_today == "Ναι":
        rainfall = st.number_input("Όγκος Βροχής Σήμερα (mm)", value=5.0, min_value=0.1)
    else:
        rainfall = 0.0

with col2:
    humidity_9am = st.slider("Υγρασία στις 9πμ (%)", 0, 100, 60)
    humidity_3pm = st.slider("Υγρασία στις 3μμ (%)", 0, 100, 50)
    pressure_9am = st.number_input("Ατμοσφαιρική Πίεση 9πμ (hPa)", value=1015.0)
    pressure_3pm = st.number_input("Ατμοσφαιρική Πίεση 3μμ (hPa)", value=1012.0)
    wind_speed = st.number_input("Ταχύτητα Ανέμου (km/h)", value=35.0)

# 4. Πρόβλεψη & Λογική
if st.button("🔮 Πρόβλεψη", use_container_width=True):

    # Γεμίζουμε τα δεδομένα με φυσιολογικές τιμές και
    # υπολογίζουμε τα Feature Engineering (TempRange, Diff) που ζητάει το μοντέλο
    base_data = {
        'MinTemp': min_temp,
        'MaxTemp': max_temp,
        'Rainfall': rainfall,
        'Evaporation': 5.0,  # Τυπική τιμή εξάτμισης
        'Sunshine': 8.0,  # Τυπική τιμή ηλιοφάνειας
        'WindGustSpeed': wind_speed,
        'WindSpeed9am': wind_speed * 0.4,
        'WindSpeed3pm': wind_speed * 0.6,
        'Humidity9am': humidity_9am,
        'Humidity3pm': humidity_3pm,
        'Pressure9am': pressure_9am,
        'Pressure3pm': pressure_3pm,
        'Cloud9am': 4.0,
        'Cloud3pm': 4.0,
        'Temp9am': min_temp + 5,
        'Temp3pm': max_temp - 2,
        'RainToday': 1 if rain_today == "Ναι" else 0,
        'Month': month,
        'TempRange': max_temp - min_temp,
        'HumidityDiff': humidity_3pm - humidity_9am,
        'PressureDiff': pressure_3pm - pressure_9am
    }

    df_input = pd.DataFrame([base_data])

    # Ευθυγράμμιση με τα features της εκπαίδευσης
    df_final = df_input.reindex(columns=feature_names, fill_value=0)

    # Scaling & Predict
    X_scaled = scaler.transform(df_final)
    prediction = model.predict(X_scaled)[0]
    prob = model.predict_proba(X_scaled)[0][1]

    # 5. Αποτελέσματα
    st.divider()

    if prob >= 0.70:
        st.error(f"### 🌧️ Πρόβλεψη: **ΒΡΟΧΗ** (Πιθανότητα: {prob:.1%})")
        # Εντυπωσιακό κεντραρισμένο Emoji βροχής
        st.markdown("<h1 style='text-align: center; font-size: 80px;'>🌧️ ⛈️ ☔</h1>", unsafe_allow_html=True)
        st.info("Υψηλή πιθανότητα βροχόπτωσης. Ετοιμαστείτε για κακοκαιρία!")

    elif prob <= 0.30:
        st.success(f"### ☀️ Πρόβλεψη: **ΛΙΑΚΑΔΑ** (Πιθανότητα βροχής: {prob:.1%})")
        # Εντυπωσιακό κεντραρισμένο Emoji ήλιου
        st.markdown("<h1 style='text-align: center; font-size: 80px;'>☀️ 🕶️ 🌞</h1>", unsafe_allow_html=True)
        st.info("Χαμηλή πιθανότητα βροχής. Υπέροχος καιρός για βόλτα!")

    else:
        st.warning(f"### ⛅ Πρόβλεψη: **ΑΣΤΑΘΕΙΑ** (Πιθανότητα βροχής: {prob:.1%})")
        # Εντυπωσιακό κεντραρισμένο Emoji συννεφιάς
        st.markdown("<h1 style='text-align: center; font-size: 80px;'>⛅ 🌥️ ☁️</h1>", unsafe_allow_html=True)
        st.info("Ο καιρός είναι ασταθής. Καλό είναι να έχετε μια ομπρέλα μαζί σας.")