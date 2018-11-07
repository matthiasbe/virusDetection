import numpy as np

def false_positive(prediction, expected):
    assert(len(prediction) == len(expected))
    n = np.sum((prediction == 1) & (expected == -1))
    return float(n)/len(prediction)
def missed_positive(prediction, expected):
    assert(len(prediction) == len(expected))
    n = np.sum((prediction == -1) & (expected == 1))
    return float(n)/len(prediction)
def correct_prediction(prediction, expected):
    assert(len(prediction) == len(expected))
    n = np.sum(prediction == expected)
    return float(n)/len(prediction)