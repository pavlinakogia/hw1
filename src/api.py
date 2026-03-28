from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import uvicorn
import os

# Δημιουργία της εφαρμογής
app = FastAPI(title="Weather Prediction API", description="API για πρόβλεψη βροχόπτωσης")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Φόρτωση του μοντέλου, του scaler και των ονομάτων των στηλών
model = joblib.load(os.path.join(BASE_DIR, "models", "best_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "models", "scaler.pkl"))
feature_names = joblib.load(os.path.join(BASE_DIR, "models", "feature_names.pkl"))

# Ορισμός της μορφής των δεδομένων εισόδου (Pydantic)
class WeatherInput(BaseModel):
    data: dict

@app.get("/")
def read_root():
    return {"message": "Το Weather API λειτουργεί! Πηγαίνετε στο /docs για δοκιμή."}

@app.post("/predict")
def predict_weather(input_data: WeatherInput):
    # 1. Μετατροπή των δεδομένων του χρήστη σε DataFrame
    df_input = pd.DataFrame([input_data.data])

    # 2. Αν υπάρχουν κατηγορικές μεταβλητές, τις κάνουμε One-Hot Encode
    df_encoded = pd.get_dummies(df_input)

    # 3. Ευθυγράμμιση με τις στήλες που "ξέρει" το μοντέλο (βάζει 0 όπου λείπουν δεδομένα)
    df_final = df_encoded.reindex(columns=feature_names, fill_value=0)

    # 4. Scaling
    X_scaled = scaler.transform(df_final)

    # 5. Πρόβλεψη και Πιθανότητα
    prediction = model.predict(X_scaled)[0]

    # Ελέγχουμε αν το μοντέλο υποστηρίζει predict_proba
    prob = None
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X_scaled)[0][1]

    # 6. Δημιουργία της απάντησης (JSON)
    result = {
        "prediction": int(prediction),
        "label": "Rain" if prediction == 1 else "No Rain"
    }
    if prob is not None:
        result["probability_of_rain"] = round(float(prob), 4)

    return result

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)