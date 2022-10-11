import json
import sqlite3

import joblib
import numpy as np
from flask import Flask, request
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras


def input_co2(input_request):
    input_co2_np = np.array(input_request["co2"])
    input_co2 = input_co2_np[:, np.newaxis]
    scaler_data = joblib.load("./model/scaler_co2.gz")
    scaler = MinMaxScaler()
    scaler.min_, scaler.scale_ = scaler_data.min_[0], scaler_data.scale_[0]
    scaled_inputs = scaler.transform(input_co2)
    scaled_inputs = scaled_inputs.reshape(1, 48, 1)
    predictions_scaled = np.array(model_co2.predict(scaled_inputs))
    predictions_descaled = scaler.inverse_transform(predictions_scaled)
    input_json = {"co2": input_co2_np.tolist()}
    return predictions_descaled, input_json


def input_temp(input_request):
    input_temp_np = np.array(input_request["temp"])
    input_temp = input_temp_np[:, np.newaxis]
    scaler_data = joblib.load("./model/scaler_temp.gz")
    scaler = MinMaxScaler()
    scaler.min_, scaler.scale_ = scaler_data.min_[0], scaler_data.scale_[0]
    scaled_inputs = scaler.transform(input_temp)
    scaled_inputs = scaled_inputs.reshape(1, 48, 1)
    predictions_scaled = np.array(model_temp.predict(scaled_inputs))
    predictions_descaled = scaler.inverse_transform(predictions_scaled)
    input_json = {"temp": input_temp_np.tolist()}
    return predictions_descaled, input_json


# def input_hum(input_request):
#    input_hum_np = np.array(input_request["hum"])
#    input_hum = input_hum_np[:, np.newaxis]
#    scaler_data = joblib.load("./model/scaler_hum.gz")
#    scaler = MinMaxScaler()
#    scaler.min_, scaler.scale_ = scaler_data.min_[0], scaler_data.scale_[0]
#    scaled_inputs = scaler.transform(input_hum)
#    scaled_inputs = scaled_inputs.reshape(1, 48, 1)
#    predictions_scaled = np.array(model_hum.predict(scaled_inputs))
#    predictions_descaled = scaler.inverse_transform(predictions_scaled)
#    input_json = {"hum": input_hum_np.tolist()}
#    return predictions_descaled, input_json


app = Flask(__name__)
with open("./model/official/co2_model.json") as f:
    model_co2 = keras.models.model_from_json(f.read())
    model_co2.load_weights("./model/official/co2_weights.h5")
with open("./model/official/temp_model.json") as f:
    model_temp = keras.models.model_from_json(f.read())
    model_temp.load_weights("./model/official/temp_weights.h5")
# with open("./model/hum_model.json") as f:
#    model_hum = keras.models.model_from_json(f.read())
#    model_hum.load_weights("./model/hum_weights.h5")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    input_request = request.json

    prediction_co2, input_json_co2 = input_co2(input_request)
    prediction_temp, input_json_temp = input_temp(input_request)
    #    prediction_hum, input_json_hum = input_hum(input_request)

    data_json_co2 = {"co2": prediction_co2.tolist()}
    data_json_temp = {"temp": prediction_temp.tolist()}
    #    data_json_hum = {"hum": prediction_hum.tolist()}

    connection = sqlite3.connect("predictions_multi.db")
    cursor = connection.cursor()
    cursor.execute(
        """INSERT INTO pred (date, input_co2, prediction_co2, input_temp, prediction_temp) VALUES (datetime('now'), ?, ?, ?, ?);""",
        [
            json.dumps(input_json_co2),
            json.dumps(data_json_co2),
            json.dumps(input_json_temp),
            json.dumps(data_json_temp),
            # json.dumps(input_json_hum),
            # json.dumps(data_json_hum),
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
            """SELECT prediction_co2, prediction_temp FROM pred ORDER BY date DESC LIMIT 1;"""
        )
        prediction = cursor.fetchall()
        connection.close()
        return prediction


@app.route("/input", methods=["GET"])
def input_p():
    if request.method == "GET":
        connection = sqlite3.connect("predictions_multi.db")
        cursor = connection.cursor()
        cursor.execute(
            """SELECT input_co2, input_temp FROM pred ORDER BY date DESC LIMIT 1;"""
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
            """SELECT prediction_co2, prediction_temp FROM pred WHERE id=(SELECT MAX(id) FROM pred) - 48;"""
        )
        real_nn = cursor.fetchall()
        connection.close()
        return real_nn


if __name__ == "__main__":
    app.run()
