from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum

import joblib
import pandas as pd
import json

from src.preprocessing import ohe_transform

app = FastAPI()


class HomeOwnership(str, Enum):
    RENT = "RENT"
    MORTGAGE = "MORTGAGE"
    OWN = "OWN"
    OTHER = "OTHER"


class LoanIntent(str, Enum):
    PERSONAL = "PERSONAL"
    EDUCATION = "EDUCATION"
    MEDICAL = "MEDICAL"
    VENTURE = "VENTURE"
    HOMEIMPROVEMENT = "HOMEIMPROVEMENT"
    DEBTCONSOLIDATION = "DEBTCONSOLIDATION"


class LoanGrade(str, Enum):
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    E = "E"
    F = "F"
    G = "G"


class CreditRiskBase(BaseModel):
    person_age: int = Field(..., description="Age of the borrower")
    person_income: int = Field(..., description="Annual income of the borrower")
    person_home_ownership: HomeOwnership = Field(
        ..., description="Home ownership status"
    )
    person_emp_length: Optional[float] = Field(
        None, description="Employment length in years"
    )
    loan_intent: LoanIntent = Field(..., description="Purpose of the loan")
    loan_grade: LoanGrade = Field(..., description="Assigned loan grade")
    loan_amnt: int = Field(..., description="Loan amount")
    loan_int_rate: Optional[float] = Field(None, description="Interest rate")
    loan_percent_income: float = Field(
        ..., description="Percent of income for the loan"
    )
    cb_person_default_on_file: str = Field(..., description="Historical default (Y/N)")
    cb_person_cred_hist_length: int = Field(..., description="Length of credit history")

    class Config:
        from_attributes = True


with open("models/best_threshold.json", "rb") as f:
    json_load = json.load(f)
    threshold = json_load["threshold"]
    best_model = json_load["model_name"]

model = joblib.load(f"models/{best_model}_best.pkl")
ohe_default = joblib.load("models/ohe_default_on_file.pkl")
ohe_home = joblib.load("models/ohe_home_ownership.pkl")
ohe_grade = joblib.load("models/ohe_loan_grade.pkl")
ohe_intent = joblib.load("models/ohe_loan_intent.pkl")


@app.post("/predict")
def predict(data: CreditRiskBase):
    input_data = pd.DataFrame([data.model_dump()])
    processed_data = ohe_transform(
        input_data, "person_home_ownership", "home_ownership", ohe_home
    )
    processed_data = ohe_transform(
        processed_data, "loan_intent", "loan_intent", ohe_intent
    )
    processed_data = ohe_transform(
        processed_data, "loan_grade", "loan_grade", ohe_grade
    )
    processed_data = ohe_transform(
        processed_data, "cb_person_default_on_file", "default_onfile", ohe_default
    )
    proba = model.predict_proba(processed_data)[:, 1][0]
    prediction = 1 if proba >= threshold else 0

    return {
        "status": "success",
        "prediction": int(prediction),
        "probability": float(proba),
    }
