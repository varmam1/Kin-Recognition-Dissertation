import numpy as np
import pytest
from ...prediction import predictor


def test_prediction_algo_1_with_Euclidean_metric_with_1_view():
    U = [np.identity(3)]
    w = np.array([1.0])

    xs = np.array([[[1, 2, 3]],
                   [[1, 1, 1]],
                   [[2, 1, 1]],
                   [[1, 2, 1]],
                   [[1, 1, 2]]])

    ys = np.array([[[2, 2, 3]],
                   [[1, 5, 1]],
                   [[4, 2, 2]],
                   [[-1, -2, -1]],
                   [[3, 1, 4]]])
    theta = 0.9

    out = predictor.predict_if_kin_1(w, U, xs, ys, theta)
    expectedOut = []
    for i in range(5):
        x = xs[i][0]
        y = ys[i][0]
        expectedOut.append((np.dot(x, np.transpose(y))/(np.linalg.norm(x) * np.linalg.norm(y)) + 1)/2 >= 0.9)
    expectedOut = np.array(expectedOut)
    assert (out == expectedOut).all()
