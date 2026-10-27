
import torch.nn as nn

class TCN(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(20,64,3,padding=1),
            nn.ReLU(),
            nn.Conv1d(64,64,3,padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        self.head = nn.Linear(64,10)

    def forward(self,x):
        x=x.transpose(1,2)
        x=self.encoder(x).squeeze(-1)
        return self.head(x)
