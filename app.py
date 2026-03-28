from fastapi import FastAPI
import joblib

app = FastAPI()

model = joblib.load("model.joblib")

@app.get("/")
def health():
    return {
        "name": "Harshitha",
        "rollno": "2022BCS0209"
    }

@app.post("/predict")
def predict(data: list):
    prediction = model.predict([data])

    return {
        "prediction": int(prediction[0]),
        "name": "Harshitha",
        "rollno": "2022BCS0209"
    }