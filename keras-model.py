import json
import sqlite3
import tensorflow.keras as keras
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request
from sklearn.preprocessing import MinMaxScaler


def input_co2(input):
    inputnp = np.array(input["co2"])
    input = inputnp[:, np.newaxis]
    scaler_data = joblib.load("./model/scaler_co2.gz")
    scaler = MinMaxScaler()
    scaler.min_, scaler.scale_ = scaler_data.min_[0], scaler_data.scale_[0]
    scaled_inputs = scaler.transform(input)
    scaled_inputs = scaled_inputs.reshape(1, 96, 1)
    predictions_descaled = np.array(model_co2.predict(scaled_inputs))
    input_json = {"input": inputnp.tolist()}
    return predictions_descaled, input_json


def input_temp(input):
    inputnp = np.array(input["temp"])
    input = inputnp[:, np.newaxis]
    scaler_data = joblib.load("./model/scaler_temp.gz")
    scaler = MinMaxScaler()
    scaler.min_, scaler.scale_ = scaler_data.min_[0], scaler_data.scale_[0]
    scaled_inputs = scaler.transform(input)
    scaled_inputs = scaled_inputs.reshape(1, 96, 1)
    predictions_descaled = np.array(model_temp.predict(scaled_inputs))
    input_json = {"input": inputnp.tolist()}
    return predictions_descaled, input_json


def input_hum(input):
    inputnp = np.array(input["hum"])
    input = inputnp[:, np.newaxis]
    scaler_data = joblib.load("./model/scaler_hum.gz")
    scaler = MinMaxScaler()
    scaler.min_, scaler.scale_ = scaler_data.min_[0], scaler_data.scale_[0]
    scaled_inputs = scaler.transform(input)
    scaled_inputs = scaled_inputs.reshape(1, 96, 1)
    predictions_descaled = np.array(model_hum.predict(scaled_inputs))
    input_json = {"input": inputnp.tolist()}
    return predictions_descaled, input_json


app = Flask(__name__)
with open("./model/co2_model.json") as f:
    model_co2 = keras.models.model_from_json(f.read())
    model_co2.load_weights("./model/co2_weights.h5")
with open("./model/temp_model.json") as f:
    model_temp = keras.models.model_from_json(f.read())
    model_temp.load_weights("./model/temp_weights.h5")
with open("./model/hum_model.json") as f:
    model_hum = keras.models.model_from_json(f.read())
    model_hum.load_weights("./model/hum_weights.h5")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    input = request.json

    prediction_co2, input_json_co2 = input_co2(input)
    prediction_temp, input_json_temp = input_temp(input)
    prediction_hum, input_json_hum = input_hum(input)

    data_json_co2 = {"co2": prediction_co2.tolist()}
    data_json_temp = {"temp": prediction_temp.tolist()}
    data_json_hum = {"hum": prediction_hum.tolist()}

    connection = sqlite3.connect("predictions_multi.db")
    cursor = connection.cursor()
    cursor.execute(
        """INSERT INTO pred (date, input_co2, prediction_co2, input_temp, prediction_temp, input_hum, prediction_hum ) VALUES (datetime('now'), ?, ?, ?, ?, ?, ?);""",
        [
            json.dumps(input_json_co2),
            json.dumps(data_json_co2),
            json.dumps(input_json_temp),
            json.dumps(data_json_temp),
            json.dumps(input_json_hum),
            json.dumps(data_json_hum),
        ],
    )
    connection.commit()
    connection.close()
    return "Done"


@app.route("/results", methods=["GET"])
def results():
    if request.method == "GET":
        connection = sqlite3.connect("predictions_multi.db")
        cursor = connection.cursor()
        cursor.execute(
            """SELECT prediction_co2, prediction_temp, prediction_hum FROM pred ORDER BY date DESC LIMIT 1;"""
        )
        prediction = cursor.fetchall()
        connection.close()
        return prediction


@app.route("/input", methods=["GET"])
def input():
    if request.method == "GET":
        connection = sqlite3.connect("predictions_multi.db")
        cursor = connection.cursor()
        cursor.execute(
            """SELECT input_co2, input_temp, input_hum FROM pred ORDER BY date DESC LIMIT 1;"""
        )
        input_nn = cursor.fetchall()
        connection.close()
        return input_nn


@app.route("/real", methods=["GET"])
def real():
    if request.method == "GET":
        connection = sqlite3.connect("predictions_multi.db")
        cursor = connection.cursor()
        cursor.execute(
            """SELECT prediction_co2, prediction_temp, prediction_hum FROM Pred WHERE id=(SELECT MAX(id) FROM Pred) - 96;"""
        )
        real_nn = cursor.fetchall()
        connection.close()
        return real_nn


if __name__ == "__main__":
    app.run()
