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
    def __init__(self, n_features, n_hidden=32, n_layers=2):
        super(TSModel, self).__init__()

        # LSTM architecture
        self.n_hidden = n_hidden
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=n_hidden,
            batch_first=True,
            num_layers=n_layers,
            dropout=0.5,
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


app = Flask(__name__)
# data_dir = os.path.join(app.instance_path, "data")
# os.makedirs(data_dir, exist_ok=True)
device = torch.device("cpu")
model = TSModel(1)
model.load_state_dict(
    torch.load("./model/model96_48_05d_0005_480b.pt", map_location=device)
)
model.eval()


@app.route("/predict", methods=["GET", "POST"])
def predict():
    input = request.json
    input = np.array(input["data"])
    input = input[:, np.newaxis]
    scaler_data = joblib.load("./model/scaler.gz")
    scaler = MinMaxScaler()
    scaler.min_, scaler.scale_ = scaler_data.min_[0], scaler_data.scale_[0]
    scaled_inputs = scaler.transform(input)
    predictions_descaled = np.array(get_prediction(scaled_inputs))

    # return jsonify({"predictions": predictions_descaled.tolist()})
    data_json = {"predictions": predictions_descaled.tolist()}
    connection = sqlite3.connect("predictions.db")
    cursor = connection.cursor()
    cursor.execute(
        """INSERT INTO pred VALUES (datetime('now'), ?);""", [json.dumps(data_json)]
    )
    connection.commit()
    connection.close()
    return "Done"


@app.route("/results", methods=["GET"])
def data():
    if request.method == "GET":
        connection = sqlite3.connect("predictions.db")
        cursor = connection.cursor()
        cursor.execute("""SELECT Prediction FROM pred ORDER BY Date DESC LIMIT 1;""")
        prediction = cursor.fetchall()
        connection.close()
        return prediction


if __name__ == "__main__":
    app.run()
