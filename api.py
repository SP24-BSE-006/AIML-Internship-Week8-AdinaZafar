import pickle
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

MODEL_PATH = 'models/titanic_pipeline.pkl'
MODEL_NAME = 'LogReg_Baseline'
MODEL_VERSION = '1.0'

app = FastAPI(title='Titanic Survival Predictor', version=MODEL_VERSION)

with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)


class PassengerInput(BaseModel):
    pclass: int = Field(..., ge=1, le=3, description='Passenger class 1-3')
    sex: str = Field(..., description='male or female')
    age: float = Field(..., ge=0, le=100)
    sibsp: int = Field(default=0, ge=0)
    parch: int = Field(default=0, ge=0)
    fare: float = Field(..., ge=0)
    embarked: str = Field(default='S', description='C, Q, or S')


class SurvivalOutput(BaseModel):
    survived: int
    probability: float
    verdict: str


class HealthOutput(BaseModel):
    model_config = {'protected_namespaces': ()}
    model_name: str
    version: str
    status: str


def engineer_features(p: PassengerInput) -> pd.DataFrame:
    if p.sex == 'male':
        title = 'Master' if p.age < 13 else 'Mr'
    else:
        title = 'Miss' if p.age < 18 else 'Mrs'

    family_size = p.sibsp + p.parch + 1
    is_alone = 1 if family_size == 1 else 0

    if p.age <= 12:
        age_group = 'Child'
    elif p.age <= 18:
        age_group = 'Teen'
    elif p.age <= 35:
        age_group = 'Young Adult'
    elif p.age <= 60:
        age_group = 'Adult'
    else:
        age_group = 'Senior'

    row = {
        'Pclass': p.pclass,
        'Sex': p.sex,
        'Age': p.age,
        'SibSp': p.sibsp,
        'Parch': p.parch,
        'Fare': p.fare,
        'Embarked': p.embarked,
        'Title': title,
        'FamilySize': family_size,
        'IsAlone': is_alone,
        'AgeGroup': age_group,
        'Deck': 'U',
    }
    return pd.DataFrame([row])


@app.get('/health', response_model=HealthOutput)
def health():
    return HealthOutput(model_name=MODEL_NAME, version=MODEL_VERSION, status='healthy')


@app.post('/predict', response_model=SurvivalOutput)
def predict(p: PassengerInput):
    df = engineer_features(p)
    pred = int(model.predict(df)[0])
    prob = float(model.predict_proba(df)[0][1])
    verdict = 'Survived' if pred == 1 else 'Did not survive'
    return SurvivalOutput(survived=pred, probability=round(prob, 4), verdict=verdict)
