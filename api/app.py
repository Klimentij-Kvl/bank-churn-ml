from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel

app = FastAPI()

model = joblib.load('../models/catboost_bank_churn.joblib')

class PredictRequest(BaseModel):
    CreditScore: int
    Geography: str
    Gender: str
    Age: int
    Tenure: int
    Balance: int
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: int

@app.post('/predict')
def predict(data: dict):
    df = pd.DataFrame([data])
    proba = model.predict_proba(df)[0][1]

    return {
        "probability": float(proba),
        "will_leave": bool(proba > 0.5)
    }