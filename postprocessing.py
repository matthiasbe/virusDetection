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

def evaluate(prediction, expected):
	print("False positive : {0:.2f}".format(false_positive(prediction, expected)))
	print("Missed positive : {0:.2f}".format(missed_positive(prediction, expected)))
	print("Correct prediction : {0:.2f}".format(correct_prediction(prediction, expected)))