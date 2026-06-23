from fastapi.testclient import TestClient
from app.api import app

client = TestClient(app)


def test_predict_young_female_first_class():
    payload = {'pclass': 1, 'sex': 'female', 'age': 22, 'sibsp': 0,
               'parch': 0, 'fare': 80.0, 'embarked': 'S'}
    r = client.post('/predict', json=payload)
    assert r.status_code == 200
    assert r.json()['survived'] == 1


def test_predict_older_male_third_class():
    payload = {'pclass': 3, 'sex': 'male', 'age': 45, 'sibsp': 0,
               'parch': 0, 'fare': 7.5, 'embarked': 'S'}
    r = client.post('/predict', json=payload)
    assert r.status_code == 200
    assert r.json()['survived'] == 0


def test_health_check():
    r = client.get('/health')
    assert r.status_code == 200
    assert r.json()['status'] == 'healthy'


def test_invalid_pclass_returns_422():
    payload = {'pclass': 5, 'sex': 'male', 'age': 30, 'fare': 10.0}
    r = client.post('/predict', json=payload)
    assert r.status_code == 422


def test_missing_required_field_returns_422():
    payload = {'pclass': 2, 'sex': 'female', 'age': 30}
    r = client.post('/predict', json=payload)
    assert r.status_code == 422