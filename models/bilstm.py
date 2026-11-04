import torch
import torch.nn as nn


class BiLSTM(nn.Module):

    def __init__(
        self,
        input_size=20,
        hidden_size=64,
        num_layers=2,
        dropout=0.1,
        output_size=10
    ):
        super().__init__()

        self.encoder = nn.LSTM(
            input_size=input_size,
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

        # x: [batch, 60, 20]

        output, _ = self.encoder(x)

        # Shared temporal representation
        # Use final temporal representation
        latent = output[:, -1, :]

        return self.head(latent)