import io
import json
import joblib

from flask import Flask, jsonify, request
import numpy as np

from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import DataLoader
from train import TimeSeriesDataset, TSModel

app = Flask(__name__)
imagenet_class_index = json.load(open("<PATH/TO/.json/FILE>/input.json"))
model = TSModel(1)
model.load_state_dict(
    torch.load("/home/kr/MachineLearning/flask-server/model/model144_2.pt")
)
model.eval()


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
    scaler = joblib.load("/home/kr/MachineLearning/flask-server/model/scaler.gz")
    descaler = MinMaxScaler()
    descaler.min_, descaler.scale_ = scaler.min_[0], scaler.scale_[0]
    predictions_descaled = descale(descaler, predictions)
    labels_descaled = descale(descaler, labels)

    return predictions_descaled, labels_descaled


def get_prediction(data):
    sequence_length = 144
    predictions_descaled, labels_descaled = prediction(data, sequence_length)
    return predictions_descaled, labels_descaled


@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        file = request.files["file"]
        data = file.read()
        predictions_descaled, labels_descaled = get_prediction(data)
        return jsonify({"predictions": predictions_descaled, "labels": labels_descaled})


if __name__ == "__main__":
    app.run()
