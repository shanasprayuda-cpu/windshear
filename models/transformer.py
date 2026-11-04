import torch
import torch.nn as nn


class Transformer(nn.Module):

    def __init__(
        self,
        input_size=20,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.1,
        output_size=10
    ):
        super().__init__()

        self.input_projection = nn.Linear(
            input_size,
            d_model
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu"
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.head = nn.Linear(
            d_model,
            output_size
        )

    def forward(self, x):

        # x: [batch, 60, 20]

        x = self.input_projection(x)

        x = self.encoder(x)

        # Shared temporal representation
        latent = x[:, -1, :]

        return self.head(latent)