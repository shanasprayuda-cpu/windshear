from models.tcn import TCN
from models.bilstm import BiLSTM
from models.transformer import Transformer
from models.tcn_bilstm import TCNBiLSTM
from models.tcn_transformer import TCNTransformer


def build_model(name):

    if name == "TCN":
        return TCN()

    elif name == "BiLSTM":
        return BiLSTM()

    elif name == "Transformer":
        return Transformer()

    elif name == "TCN-BiLSTM":
        return TCNBiLSTM()

    elif name == "TCN-Transformer":
        return TCNTransformer()

    raise ValueError(
        f"Unknown architecture: {name}"
    )