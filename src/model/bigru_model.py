import torch
import torch.nn as nn

class AffectiveBiGRU(nn.Module):
    def __init__(self, input_size=41, hidden_size=64, num_layers=2, num_classes=3, dropout=0.3):
        super(AffectiveBiGRU, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Bidirectional GRU Layer
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # Output Regularization & Classification Head
        self.batch_norm = nn.BatchNorm1d(hidden_size * 2)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        gru_out, _ = self.gru(x)  # (batch_size, seq_len, hidden_size * 2)
        
        # Extract last time-step feature vector
        out = gru_out[:, -1, :]    # (batch_size, hidden_size * 2)
        
        out = self.batch_norm(out)
        out = self.dropout(out)
        logits = self.fc(out)      # (batch_size, num_classes)
        return logits