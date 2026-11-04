import torch
import torch.nn as nn


class TCNTransformer(nn.Module):

    def __init__(
        self,
        input_size=20,
        tcn_channels=64,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
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

        self.projection = nn.Linear(
            tcn_channels,
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

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.head = nn.Linear(
            d_model,
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

        x = self.projection(x)

        # [batch, 60, 64]

        x = self.transformer(x)

        latent = x[:, -1, :]

        return self.head(latent)