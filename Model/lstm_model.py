"""
LSTM Model with RevIN preprocessing for BTP prediction.

Architecture:
    Input (raw) → RevIN(norm) → LSTM → FC_btp → BTP
                            → FC_thermal → Thermal
"""
import torch
import torch.nn as nn
from revin import RevIN


class LSTMModel(nn.Module):

    def __init__(self, input_size, hidden_size=64, num_layers=2,
                 thermal_output_size=6):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.thermal_output_size = thermal_output_size

        self.revin = RevIN(input_size, eps=1e-5, affine=True)

        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True
        )

        self.fc_btp = nn.Linear(hidden_size, 1)
        self.fc_thermal = nn.Linear(hidden_size, thermal_output_size)

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, input_size) — raw physical values
        Returns:
            btp_pred: (batch_size, 1)
            thermal_pred: (batch_size, thermal_output_size)
        """
        x_norm = self.revin(x, mode='norm')

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size,
                         device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size,
                         device=x.device)
        out, (_, _) = self.lstm(x_norm, (h0, c0))
        last_out = out[:, -1, :]

        btp_pred = self.fc_btp(last_out)
        thermal_pred = self.fc_thermal(last_out)

        return btp_pred, thermal_pred
