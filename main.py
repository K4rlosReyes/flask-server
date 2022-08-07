import json
import os

import joblib
import numpy as np
import pandas as pd
import torch
from flask import Flask, jsonify, request
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader
from werkzeug.utils import secure_filename
from core.train import TimeSeriesDataset, TSModel

app = Flask(__name__)
data_dir = os.path.join(app.instance_path, "data")
os.makedirs(data_dir, exist_ok=True)
model = TSModel(1)
model.load_state_dict(torch.load("./model/model144_2.pt"))
model.eval()


def rescale_data(scaler, df):
    """Rescale all features using MinMaxScaler() to same scale, between 0 and 1."""

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

    test_dataset = TimeSeriesDataset(
        np.array(df), np.array(df), seq_len=sequence_length
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    predictions = []
    labels = []
    with torch.no_grad():
        for features, target in test_loader:
            features = torch.Tensor(np.array(features))
            output = model(features)
            predictions.append(output.tolist())
            labels.append(target.tolist())

    # Bring predictions back to original scale
    scaler = joblib.load("model/scaler.gz")
    descaler = MinMaxScaler()
    descaler.min_, descaler.scale_ = scaler.min_[0], scaler.scale_[0]
    predictions_descaled = descale(descaler, predictions)
    labels_descaled = descale(descaler, labels)

    return predictions_descaled, labels_descaled


def get_prediction(data):
    sequence_length = 144
    predictions_descaled, labels_descaled = prediction(data, sequence_length)
    return predictions_descaled, labels_descaled


@app.route("/predict", methods=["GET"])
def predict():
    if request.method == "GET":
        input = data()
        predictions_descaled, labels_descaled = get_prediction(input)
        return jsonify({"predictions": predictions_descaled, "labels": labels_descaled})


@app.route("/data", methods=["POST"])
def data():
    if request.method == "POST":
        file = request.json
        # file.save(os.path.join(data_dir, secure_filename(file.filename)))
        return file["data"]


if __name__ == "__main__":
    app.run()
