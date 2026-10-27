import torch
import torch.nn as nn


class TCNBiLSTM(nn.Module):

    def __init__(
        self,
        input_size=20,
        tcn_channels=64,
        hidden_size=64,
        num_layers=2,
        dropout=0.1,
        output_size=10
    ):
        super().__init__()

        self.tcn = nn.Sequential(

            nn.Conv1d(
                input_size,
                tcn_channels,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.Conv1d(
                tcn_channels,
                tcn_channels,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU()
        )

        self.bilstm = nn.LSTM(
            input_size=tcn_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.head = nn.Linear(
            hidden_size * 2,
            output_size
        )

    def forward(self, x):

        # x:
        # [batch, 60, 20]

        x = x.transpose(1, 2)

        # [batch, 20, 60]

        x = self.tcn(x)

        # [batch, 64, 60]

        x = x.transpose(1, 2)

        # [batch, 60, 64]

        x, _ = self.bilstm(x)

        # Final temporal representation

        latent = x[:, -1, :]

        return self.head(latent)