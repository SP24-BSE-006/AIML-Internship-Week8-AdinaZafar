# integration test log — step 15

5 curl requests run against the live FastAPI server.

---

### request A — 1st class female, embarked C
```
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"pclass":1,"sex":"female","age":22,"sibsp":1,"parch":0,"fare":151.55,"embarked":"C"}'
```
response: `{"survived":1,"probability":0.9566,"verdict":"Survived"}`

---

### request B — 3rd class male, elderly
```
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"pclass":3,"sex":"male","age":65,"sibsp":0,"parch":0,"fare":7.75,"embarked":"Q"}'
```
response: `{"survived":0,"probability":0.0527,"verdict":"Did not survive"}`

---

### request C — 2nd class boy with family
```
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"pclass":2,"sex":"male","age":8,"sibsp":1,"parch":2,"fare":26.0,"embarked":"S"}'
```
response: `{"survived":1,"probability":0.6783,"verdict":"Survived"}`

the Title feature maps age < 13 male to "Master" which had a higher survival rate than "Mr" in training data, so the model picked this up correctly.

---

### request D — 3rd class woman, large family
```
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"pclass":3,"sex":"female","age":35,"sibsp":4,"parch":2,"fare":31.275,"embarked":"S"}'
```
response: `{"survived":0,"probability":0.1626,"verdict":"Did not survive"}`

---

### request E — 1st class young male, high fare (surprising case)
```
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"pclass":1,"sex":"male","age":28,"sibsp":0,"parch":0,"fare":263.0,"embarked":"S"}'
```
response: `{"survived":0,"probability":0.444,"verdict":"Did not survive"}`

i expected this to predict survival given 1st class + very high fare but it didn't. being male is basically overriding everything else here which lines up with what the SHAP plots showed — Sex/Title dominate the model regardless of what other features say.

