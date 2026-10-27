
import numpy as np
from sklearn.metrics import average_precision_score


def mean_auprc(y,p):

    scores=[]

    for i in range(y.shape[1]):
        scores.append(
            average_precision_score(
                y[:,i],
                p[:,i]
            )
        )

    return float(np.mean(scores))
