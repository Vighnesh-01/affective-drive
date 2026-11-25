import os, sys
# Ensure the project root (parent of 'src/') is on sys.path so
# 'from src.…' imports work regardless of the working directory.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.dataset.gru_dataset import DrivingSequenceDataset
from src.model.bigru_model import AffectiveBiGRU

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Training BiGRU Model on Device: {device}")

    # Hyperparameters
    BATCH_SIZE = 16
    EPOCHS = 35
    LEARNING_RATE = 1e-3
    SEQ_LEN = 30

    # Dataset & Loader Initialization
    dataset = DrivingSequenceDataset(csv_file="data/phase2_dataset.csv", seq_len=SEQ_LEN)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = AffectiveBiGRU(input_size=dataset.num_features, hidden_size=64, num_classes=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    # Training Loop
    model.train()
    for epoch in range(EPOCHS):
        running_loss = 0.0
        correct = 0
        total = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * X_batch.size(0)
            _, predicted = torch.max(outputs, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = (correct / total) * 100.0

        if (epoch + 1) % 5 == 0 or epoch == EPOCHS - 1:
            print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.2f}%")

    # Save Model Weights
    output_path = "data/bigru_affective.pth"
    torch.save(model.state_dict(), output_path)
    print(f"\n[SUCCESS] Model weights saved successfully to {output_path}")

if __name__ == "__main__":
    train()