import json
import sqlite3

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from flask import Flask, request
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset


class TimeSeriesDataset(Dataset):
    def __init__(self, X, seq_len=96):
        self.X = X
        self.seq_len = seq_len

    def __len__(self):
        return self.X.__len__()

    # Create Window
    def __getitem__(self, index):
        return self.X[0 : self.seq_len]


class TSModel(nn.Module):
    def __init__(self, n_features, n_hidden=64, n_layers=2):
        super(TSModel, self).__init__()

        # LSTM architecture
        self.n_hidden = n_hidden
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=n_hidden,
            batch_first=True,
            num_layers=n_layers,
            dropout=0.2,
        )
        self.linear = nn.Linear(n_hidden, 48)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        lstm_out = hidden[-1]  # Output last hidden state output
        y_pred = self.linear(lstm_out)

        return y_pred


def rescale_data(scaler, df):
    """Rescale all features using MinMaxScaler()
    to same scale, between 0 and 1."""
    df_scaled = pd.DataFrame(scaler.transform(df), index=df.index, columns=df.columns)

    return df_scaled


def descale(descaler, values):
    final = []
    for prediction in values:
        x = descaler.inverse_transform(prediction)
        final.append(x)
    return final


def prediction(df, sequence_length):
    """Make predictions."""
    test_dataset = TimeSeriesDataset(np.array(df), seq_len=sequence_length)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    predictions = []
    with torch.no_grad():
        for features in test_loader:
            features = torch.Tensor(np.array(features))
            output = model(features)
            predictions.append(output.tolist())
            break

    # Bring predictions back to original scale
    scaler = joblib.load("./model/scaler.gz")
    descaler = MinMaxScaler()
    descaler.min_, descaler.scale_ = scaler.min_[0], scaler.scale_[0]
    predictions_descaled = descale(descaler, predictions)

    return predictions_descaled


def get_prediction(data):
    sequence_length = 96
    predictions_descaled = prediction(data, sequence_length)
    return predictions_descaled


def input_co2(input):
    inputnp = np.array(input["co2"])
    input = inputnp[:, np.newaxis]
    scaler_data = joblib.load("./model/scaler_co2.gz")
    scaler = MinMaxScaler()
    scaler.min_, scaler.scale_ = scaler_data.min_[0], scaler_data.scale_[0]
    scaled_inputs = scaler.transform(input)
    predictions_descaled = np.array(get_prediction(scaled_inputs))
    input_json = {"input": inputnp.tolist()}
    return predictions_descaled, input_json


def input_temp(input):
    inputnp = np.array(input["temp"])
    input = inputnp[:, np.newaxis]
    scaler_data = joblib.load("./model/scaler_temp.gz")
    scaler = MinMaxScaler()
    scaler.min_, scaler.scale_ = scaler_data.min_[0], scaler_data.scale_[0]
    scaled_inputs = scaler.transform(input)
    predictions_descaled = np.array(get_prediction(scaled_inputs))
    input_json = {"input": inputnp.tolist()}
    return predictions_descaled, input_json


def input_hum(input):
    inputnp = np.array(input["hum"])
    input = inputnp[:, np.newaxis]
    scaler_data = joblib.load("./model/scaler_hum.gz")
    scaler = MinMaxScaler()
    scaler.min_, scaler.scale_ = scaler_data.min_[0], scaler_data.scale_[0]
    scaled_inputs = scaler.transform(input)
    predictions_descaled = np.array(get_prediction(scaled_inputs))
    input_json = {"input": inputnp.tolist()}
    return predictions_descaled, input_json


app = Flask(__name__)

device = torch.device("cpu")
model_co2 = TSModel(1)
model_temp = TSModel(1)
model_hum = TSModel(1)

model_co2.load_state_dict(torch.load("./model/modeltest50.pt", map_location=device))
model_temp.load_state_dict(torch.load("./model/modeltest50.pt", map_location=device))
model_hum.load_state_dict(torch.load("./model/modeltest50.pt", map_location=device))

model_co2.eval()
model_temp.eval()
model_hum.eval()


@app.route("/predict", methods=["GET", "POST"])
def predict():
    input = request.json

    prediction_co2, input_json_co2 = input_co2(input)
    prediction_temp, input_json_temp = input_temp(input)
    prediction_hum, input_json_hum = input_hum(input)

    data_json_co2 = {"co2": prediction_co2.tolist()}
    data_json_temp = {"temp": prediction_temp.tolist()}
    data_json_hum = {"hum": prediction_hum.tolist()}

    connection = sqlite3.connect("predictions.db")
    cursor = connection.cursor()
    cursor.execute(
        """INSERT INTO pred (Date, input_co2, prediction_co2, input_temp, prediction_temp, input_hum, prediction_hum ) VALUES (datetime('now'), ?, ?, ?, ?, ?, ?);""",
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
        connection = sqlite3.connect("predictions.db")
        cursor = connection.cursor()
        cursor.execute(
            """SELECT prediction_co2, prediction_temp, prediction_hum FROM pred ORDER BY Date DESC LIMIT 1;"""
        )
        prediction = cursor.fetchall()
        connection.close()
        return prediction


@app.route("/input", methods=["GET"])
def input():
    if request.method == "GET":
        connection = sqlite3.connect("predictions.db")
        cursor = connection.cursor()
        cursor.execute(
            """SELECT input_co2, input_temp, input_hum FROM pred ORDER BY Date DESC LIMIT 1;"""
        )
        input_nn = cursor.fetchall()
        connection.close()
        return input_nn


@app.route("/real", methods=["GET"])
def real():
    if request.method == "GET":
        connection = sqlite3.connect("predictions.db")
        cursor = connection.cursor()
        cursor.execute(
            """SELECT prediction_co2, prediction_temp, prediction_hum FROM Pred WHERE Id=(SELECT MAX(Id) FROM Pred) - 96;"""
        )
        real_nn = cursor.fetchall()
        connection.close()
        return real_nn


if __name__ == "__main__":
    app.run()
