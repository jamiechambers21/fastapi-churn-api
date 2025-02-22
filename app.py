from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()
model = joblib.load("churn_model.pkl")

@app.get("/")
def home():
    return {"message": "Customer Churn Prediction API"}

@app.post("/predict/")
def predict(data: dict):
    df = pd.DataFrame([data])
    prediction = model.predict(df)[0]
    return {"churn_prediction": int(prediction)}
