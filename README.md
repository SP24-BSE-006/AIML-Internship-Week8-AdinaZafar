# Titanic Survival Predictor — Week 8 Capstone
**Student:** Adina Zafar | SP24-BSE-006
**Internship:** AI/ML Internship Program — Week 8 of 8
**Instructor:** Zain Ul Abideen

## Project Overview
this is the capstone project for the 8-week AI/ML internship. the goal was to take everything from weeks 1–7 and ship it as an actual working system, not just a notebook. the task was to predict whether a Titanic passenger survived based on their ticket details, but the real point wasn't the prediction. it was building the full pipeline around it.

what i built: a logistic regression model wrapped in a sklearn Pipeline with custom feature engineering, tracked across 3 algorithm experiments using MLflow, exposed as a REST API via FastAPI, visualised in a 4-page Streamlit dashboard, and containerised with Docker so the whole thing runs in one command.

why Titanic: it's small enough to iterate fast but has enough messy real-world problems (missing values, categorical encoding, feature interactions) to make the engineering interesting. the domain knowledge also transfers directly, title extraction from names, family size, cabin deck. these aren't arbitrary transforms, they reflect what actually determined survival in 1912.

how it works: raw CSV goes through a ColumnTransformer pipeline that imputes, encodes, and scales, then into the classifier. the trained pipeline is serialised with pickle and loaded by the FastAPI server at startup. the Streamlit UI calls the API on every prediction request, so the model is never directly exposed to the frontend.


## Dataset Description

- **source:** Titanic — Machine Learning from Disaster (Kaggle)
- **size:** 891 rows, 11 raw features → 20 engineered features after preprocessing
- **target:** `Survived` — binary (0 = did not survive, 1 = survived)
- **split:** 80/20 stratified train/validation (random_state=42)

| Feature | Type | Notes |
|---|---|---|
| Pclass | int | passenger class 1/2/3 |
| Sex | string | male / female |
| Age | float | ~177 missing values, median imputed |
| SibSp | int | siblings/spouses aboard |
| Parch | int | parents/children aboard |
| Fare | float | log-right-skewed, scaled |
| Embarked | string | C / Q / S, 2 missing |
| Cabin | string | ~687 missing — deck letter extracted, rest marked 'U' |
| Name | string | title extracted then dropped |

**engineered features added:** Title, FamilySize, IsAlone, AgeGroup, Deck

## Architecture

```
train.csv
    │
    ▼
ColumnTransformer Pipeline
(impute → encode → scale)
    │
    ▼
LogisticRegression (champion)
    │
    ▼
titanic_pipeline.pkl
    │
    ▼
FastAPI  ──────────────────────►  /health
(api.py)                          /predict (POST)
    │
    ▼
Streamlit Dashboard
(streamlit_app.py)
├── Home
├── Predict
├── EDA
└── Model Insights (SHAP)
    │
    ▼
Docker + docker-compose
(api container :8000 + ui container :8501)
```

---

## Quick Start

```bash
git clone https://github.com/SP24-BSE-006/AIML-Internship-Week8-AdinaZafar.git
cd AIML-Internship-Week8-AdinaZafar

download `train.csv` from [kaggle.com/c/titanic/data](https://www.kaggle.com/c/titanic/data) and place it in `data/train.csv`.

then:

```bash
docker-compose up --build

- Streamlit UI → http://localhost:8501
- FastAPI docs → http://localhost:8000/docs
- MLflow UI → run `mlflow ui` separately → http://localhost:5000

---

## API Documentation

base URL: `http://localhost:8000`

| Endpoint | Method | Input | Output |
|---|---|---|---|
| `/health` | GET | — | `model_name`, `version`, `status` |
| `/predict` | POST | PassengerInput (see below) | `survived`, `probability`, `verdict` |

**PassengerInput schema:**

| Field | Type | Constraints |
|---|---|---|
| pclass | int | 1, 2, or 3 |
| sex | string | "male" or "female" |
| age | float | 0–100 |
| sibsp | int | ≥ 0 (default 0) |
| parch | int | ≥ 0 (default 0) |
| fare | float | ≥ 0 |
| embarked | string | "C", "Q", or "S" (default "S") |

**SurvivalOutput schema:**

| Field | Type | Example |
|---|---|---|
| survived | int | 1 |
| probability | float | 0.9273 |
| verdict | string | "Survived" |

invalid inputs return HTTP 422 with a validation error message.


## Model Performance Summary

three algorithms were trained and tracked in MLflow on the same stratified 80/20 split:

| Model | Accuracy | Macro F1 | ROC-AUC | Train time |
|---|---|---|---|---|
| **LogReg_Baseline** | **84.36%** | **0.8320** | **0.8725** | < 1s |
| GradientBoosting_Best | 83.24% | 0.8176 | 0.8668 | ~15s (40 GridSearchCV fits) |
| RandomForest_v1 | 80.45% | 0.7849 | 0.8401 | ~3s |

**champion: Logistic Regression** — selected programmatically by macro F1. with only 712 training rows and 34 one-hot encoded columns after preprocessing, the well-regularised linear model consistently outperformed both tree ensembles on this split. GridSearchCV ran 40 fits on GradientBoosting and still didn't close the gap.

---

## SHAP Insights

Title_Mr is the single strongest predictor by mean |SHAP value| — being an adult male overrides almost every other feature, which directly reflects the "women and children first" evacuation pattern. Fare and Pclass_3 rank second and third, confirming that class and wealth bought real survival advantage beyond just correlated demographics. one notable finding is that the engineered Title feature contributes independently of the raw Sex columns — it captures age information (Master vs Mr) that Sex alone can't, which justifies keeping both in the final feature set.

---

## Model Card

**model name:** titanic_survival_logreg_v1
**version:** 1.0
**author:** Adina Zafar | SP24-BSE-006

### Intended Use
demonstration and learning purposes only. this model is a capstone project submission for an AI/ML internship program. it is not intended for any real-world safety, legal, or decision-making use.

### Training Data
1912 Titanic passenger manifest — 891 passengers, sourced from the Kaggle Titanic competition dataset. 80% used for training (712 rows), 20% held out for validation (179 rows), stratified by survival label.

### Evaluation Results

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Did not survive (0) | 0.86 | 0.88 | 0.87 |
| Survived (1) | 0.82 | 0.79 | 0.80 |
| **Macro avg** | **0.84** | **0.84** | **0.83** |

overall accuracy: **84.36%** | ROC-AUC: **0.8725** on 20% validation split.

### Ethical Considerations
the model inherits the social structure of 1912 — women and first-class passengers had significantly higher survival rates not because of any predictive relationship worth generalising, but because of who got lifeboats first. Sex and Pclass are the two strongest signals in the data, which means the model is essentially encoding historical class and gender inequality as learned decision boundaries. this is expected and documented, but means the model should never be used to draw conclusions about any real person or group.

### Known Limitations
- dataset is very small (891 rows) — results are split-sensitive and shouldn't be treated as stable performance estimates
- Cabin data was ~77% missing, so Deck is mostly 'U' (unknown) and contributes little real signal
- the feature engineering pipeline infers Title from sex + age when Name isn't available (in the API), which is a simplification
- model reflects 1912 demographics — survival patterns from this event do not generalise to any other domain

## Tools

Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, MLflow, FastAPI, Uvicorn, Streamlit, SHAP, Pydantic, Docker, docker-compose
