import numpy as np
from avcl.metrics import average_accuracy, forgetting, backward_transfer

def test_continual_metrics():
    matrix = np.array([[.8, np.nan, np.nan], [.6, .9, np.nan], [.5, .85, .7]])
    assert abs(average_accuracy(matrix, 2) - (.5+.85+.7)/3) < 1e-8
    assert forgetting(matrix, 2) > 0
    assert backward_transfer(matrix, 2) < 0
