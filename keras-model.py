import json
import sqlite3
import tensorflow.keras as keras
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)
with open("./model/co2_model.json") as f:
    model = keras.models.model_from_json(f.read())
    model.load_weights("./model/co2_weights.h5")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    input = request.json
    inputnp = np.array(input["data"])
    input = inputnp[:, np.newaxis]
    scaler_data = joblib.load("./model/scaler.gz")
    scaler = MinMaxScaler()
    scaler.min_, scaler.scale_ = scaler_data.min_[0], scaler_data.scale_[0]
    scaled_inputs = np.array(scaler.transform(input))
    scaled_inputs = scaled_inputs.reshape(1, 96, 1)
    predictions = np.array(model.predict(scaled_inputs))

    predictions_descaled = scaler.inverse_transform(predictions)
    input_json = {"input": inputnp.tolist()}
    data_json = {"predictions": predictions_descaled.tolist()}
    connection = sqlite3.connect("predictions.db")
    cursor = connection.cursor()
    cursor.execute(
        """INSERT INTO pred (Date, Input, Prediction) VALUES (datetime('now'), ?, ?);""",
        [json.dumps(input_json), json.dumps(data_json)],
    )
    connection.commit()
    connection.close()
    return "Done"


@app.route("/results", methods=["GET"])
def results():
    if request.method == "GET":
        connection = sqlite3.connect("predictions.db")
        cursor = connection.cursor()
        cursor.execute("""SELECT Prediction FROM pred ORDER BY Date DESC LIMIT 1;""")
        prediction = cursor.fetchall()
        connection.close()
        return prediction


@app.route("/input", methods=["GET"])
def input():
    if request.method == "GET":
        connection = sqlite3.connect("predictions.db")
        cursor = connection.cursor()
        cursor.execute("""SELECT Input FROM pred ORDER BY Date DESC LIMIT 1;""")
        input_nn = cursor.fetchall()
        connection.close()
        return input_nn


@app.route("/real", methods=["GET"])
def real():
    if request.method == "GET":
        connection = sqlite3.connect("predictions.db")
        cursor = connection.cursor()
        cursor.execute(
            """SELECT Prediction FROM Pred WHERE Id=(SELECT MAX(Id) FROM Pred) - 96;"""
        )
        real_nn = cursor.fetchall()
        connection.close()
        return real_nn


if __name__ == "__main__":
    app.run()
