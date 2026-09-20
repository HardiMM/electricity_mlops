from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import tensorflow as tf

app = FastAPI(title="Electricity Consumption Prediction API")

# Load best model
model = tf.keras.models.load_model("models/best_model.keras")


class PredictionInput(BaseModel):
    sequence: list


@app.get("/")
def home():
    return {
        "message": "Electricity Consumption Prediction API is running"
    }


@app.post("/predict")
def predict(data: PredictionInput):

    sequence = np.array(data.sequence, dtype=np.float32)

    # Expected shape: 30 time steps × 15 features
    sequence = np.expand_dims(sequence, axis=0)

    prediction = model.predict(sequence, verbose=0)

    return {
        "prediction": float(prediction[0][0])
    }