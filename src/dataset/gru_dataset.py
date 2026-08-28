import pandas as pd
import torch
from torch.utils.data import Dataset

class DrivingSequenceDataset(Dataset):
    def __init__(self, csv_file="data/phase2_dataset.csv", seq_len=30):
        self.seq_len = seq_len
        self.sequences = []
        self.labels = []
        
        # Load logged feature dataset
        df = pd.read_csv(csv_file)
        
        # Exclude non-feature columns
        ignore_cols = ["sequence_id", "timestamp", "label"]
        feature_cols = [col for col in df.columns if col not in ignore_cols]
        self.num_features = len(feature_cols)
        
        # Group features into sequences
        grouped = df.groupby("sequence_id")
        
        for seq_id, group in grouped:
            feats = group[feature_cols].values
            # Majority label per sequence block
            label = int(group["label"].mode()[0])
            
            # Pad or truncate sequence to match target length
            if len(feats) >= self.seq_len:
                feats = feats[:self.seq_len]
            else:
                pad_length = self.seq_len - len(feats)
                zero_padding = [[0.0] * self.num_features] * pad_length
                feats = list(feats) + zero_padding
                
            self.sequences.append(feats)
            self.labels.append(label)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        x = torch.tensor(self.sequences[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y