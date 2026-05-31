from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)

app = FastAPI()

model = joblib.load('models/catboost_bank_churn.joblib')

class PredictRequest(BaseModel):
    CreditScore: int
    Geography: str
    Gender: str
    Age: int
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float

class PredictResponse(BaseModel):
    churn_probability: float
    will_churn: bool

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post('/predict')
def predict(data: PredictRequest):
    try:
        df = pd.DataFrame([data.model_dump()])
        proba = model.predict_proba(df)[0][1]

        return PredictResponse(churn_probability=round(float(proba), 4), will_churn=proba > 0.5)
    except Exception as e:
        logger.error(f"failed to load model: {e}")
        raise
    