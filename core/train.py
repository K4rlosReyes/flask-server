""" Module for model training. """

import argparse

import numpy as np
import pandas as pd
import seaborn as sns
import yaml

sns.set()
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

os.chdir("/home/kr/MachineLearning/carbon-forecast/core/")
import preprocess

with open("../models/params.yaml", "r") as params_file:
    params = yaml.safe_load(params_file)

model_dir = params["model_dir"]


class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, seq_len=144):
        self.X = X
        self.y = y
        self.seq_len = seq_len

    def __len__(self):
        return self.X.__len__() - 156

    # Create Window and Horizon
    def __getitem__(self, index):
        return (
            self.X[index : index + self.seq_len],
            self.y[index + self.seq_len : index + self.seq_len + 12],
        )


class TSModel(nn.Module):
    def __init__(self, n_features, n_hidden=64, n_layers=3):
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
        self.linear = nn.Linear(n_hidden, 12)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        lstm_out = hidden[-1]  # Output last hidden state output
        y_pred = self.linear(lstm_out)

        return y_pred


def train_model(
    train_df, test_df, label_name, sequence_length, batch_size, n_epochs, n_epochs_stop
):
    """Train LSTM model."""
    print("Starting with model training...")

    # Create dataloaders
    train_dataset = TimeSeriesDataset(
        np.array(train_df), np.array(train_df[label_name]), seq_len=sequence_length
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_dataset = TimeSeriesDataset(
        np.array(test_df), np.array(test_df[label_name]), seq_len=sequence_length
    )
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)
    # Set up training
    n_features = train_df.shape[1]
    model = TSModel(n_features)
    
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_hist = []
    test_hist = []

    # Start training
    best_loss = np.inf
    epochs_no_improve = 0
    for epoch in range(1, n_epochs + 1):
        running_loss = 0
        model.train()
        
        for batch_idx, (data, target) in enumerate(train_loader, 0):
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + Backward + Optimize
            data = torch.Tensor(np.array(data))
            writer.add_graph(model, data)
            writer.flush()
            output = model(data)
            loss = criterion(output, target.type_as(output))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        running_loss /= len(train_loader)
        train_hist.append(running_loss)

        # Test loss
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for data, target in test_loader:
                data = torch.Tensor(np.array(data))
                output = model(data)
                loss = criterion(output, target.type_as(output))
                test_loss += loss.item()
            test_loss /= len(test_loader)
            test_hist.append(test_loss)

            # Early stopping
            if test_loss < best_loss:
                best_loss = test_loss
                torch.save(model.state_dict(), Path(model_dir, "model144_2.pt"))
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            if epochs_no_improve == n_epochs_stop:
                print("Early stopping.")
                break

        print(
            f"Epoch {epoch} train loss: {round(running_loss,4)} test loss: {round(test_loss,4)}"
        )

        hist = pd.DataFrame()
        hist["training_loss"] = train_hist
        hist["test_loss"] = test_hist

    print("Completed.")

    return hist


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sequence-length", type=int, default=params["sequence_length"]
    )
    parser.add_argument("--batch-size", type=int, default=params["batch_size"])
    parser.add_argument("--n-epochs", type=int, default=params["n_epochs"])
    parser.add_argument("--n-epochs-stop", type=int, default=params["n_epochs_stop"])
    args = parser.parse_args()

    # Load the data and take 1 sample every 30 min
    data = preprocess.load_data("IUMPA.csv")
    data = data[0::30]

    # Scale and split the data
    train_df, test_df = preprocess.prep_data(df=data, train_frac=0.80, plot_df=False)

    # Declaring training variables
    label_name = "carbono"
    sequence_length = 144
    batch_size = 256
    n_epochs = 100
    n_epochs_stop = 5

    # Start training
    hist = train_model(
        train_df,
        test_df,
        label_name,
        sequence_length,
        batch_size,
        n_epochs,
        n_epochs_stop,
    )
    writer.close()
    