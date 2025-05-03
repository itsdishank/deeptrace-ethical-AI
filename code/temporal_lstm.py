import torch
import torch.nn as nn

class TemporalModel(nn.Module):
    def __init__(self, input_dim=1280, hidden_dim=128, num_layers=1):
        super(TemporalModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]
        prob = torch.sigmoid(self.fc(last_out))
        return prob
