import os, sys
# Ensure the project root (parent of 'src/') is on sys.path so
# 'from src.…' imports work regardless of the working directory.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from src.model.bigru_model import AffectiveBiGRU

class AffectiveDataset(Dataset):
    def __init__(self, csv_path, seq_len=30):
        df = pd.read_csv(csv_path)

        # 1. Map string labels ('NORMAL', 'ANXIOUS', 'DISTRACTED') to integer IDs (0, 1, 2)
        label_map = {"NORMAL": 0, "ANXIOUS": 1, "DISTRACTED": 2}
        if df["label"].dtype == object or isinstance(df["label"].iloc[0], str):
            df["label"] = df["label"].astype(str).str.upper().map(label_map).fillna(0)
        
        df["label"] = df["label"].astype(int)

        # 2. Extract feature columns and labels as clean NumPy arrays
        feature_cols = [c for c in df.columns if c not in ["sequence_id", "label", "timestamp"]]
        features = df[feature_cols].values.astype(np.float32)
        labels = df["label"].values.astype(np.int64)

        # 3. Create 30-frame sliding sequences
        self.X, self.y = [], []
        for i in range(len(features) - seq_len + 1):
            self.X.append(features[i:i + seq_len])
            self.y.append(labels[i + seq_len - 1])

        # 4. Convert NumPy arrays directly to PyTorch tensors
        self.X = torch.tensor(np.array(self.X, dtype=np.float32), dtype=torch.float32)
        self.y = torch.tensor(np.array(self.y, dtype=np.int64), dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def train_model():
    data_path = "data/phase2_dataset.csv"
    save_path = "data/bigru_affective.pth"

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"[ERROR] Dataset not found at {data_path}.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = AffectiveDataset(data_path, seq_len=30)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = AffectiveBiGRU(input_size=41, hidden_size=64, num_classes=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()
    epochs = 20
    print(f"[TRAINING] Starting BiGRU training for {epochs} epochs on {device}...")

    for epoch in range(epochs):
        total_loss = 0.0
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}")

    os.makedirs("data", exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"[SUCCESS] Model weights saved to {save_path}")

if __name__ == "__main__":
    train_model()