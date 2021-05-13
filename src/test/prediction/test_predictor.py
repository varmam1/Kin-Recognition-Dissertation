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

    out = predictor.predict(w, U, xs, ys, theta)
    expectedOut = []
    for i in range(5):
        x = xs[i][0]
        y = ys[i][0]
        expectedOut.append((np.dot(x, np.transpose(y))/(np.linalg.norm(x) * np.linalg.norm(y)) + 1)/2 >= theta)
    expectedOut = np.array(expectedOut)
    assert (out == expectedOut).all()


def test_prediction_algo_1_with_nonEuclidean_metric_with_2_views():
    U_1 = np.array([[1], [0], [1]])
    U_2 = np.array([[0], [1], [1]])
    U = [U_1, U_2]
    w = np.array([0.3, 0.7])

    xs = np.array([[[1, 2, 3], [2, 3, 4]],
                   [[1, 1, 1], [1, 1, 1]],
                   [[2, 1, 1], [2, 2, 1]],
                   [[1, 2, 1], [1, 2, 2]],
                   [[1, 1, 2], [3, 1, 3]]])

    ys = np.array([[[2, 2, 3], [2, 3, 3]],
                   [[1, 5, 1], [5, 5, 1]],
                   [[4, 2, 2], [10, -5, -3]],
                   [[-1, -2, -1], [-1, -2, -2]],
                   [[3, 1, 4], [1, 4, 1]]])
    theta = 0.9

    out = predictor.predict(w, U, xs, ys, theta)
    expectedOut = []

    for i in range(5):
        simScore = 0
        for j in range(2):
            A_p = np.dot(U[j], np.transpose(U[j]))
            x = xs[i][j]
            y = ys[i][j]
            top = np.dot(np.dot(x, A_p), y)
            bottom = np.sqrt(np.dot(np.dot(x, A_p), x)) * np.sqrt(np.dot(np.dot(y, A_p), y))
            simScore = simScore + w[j] * 0.5 * (top/bottom + 1)
        expectedOut.append(simScore >= theta)
    expectedOut = np.array(expectedOut)
    assert (out == expectedOut).all()


def test_prediction_algo_1_with_Euclidean_metric_with_1_view_with_tri_relationship():
    U = [np.identity(3)]
    w = np.array([1.0])

    xs = np.array([[[1, 2, 3]],
                   [[1, 1, 1]],
                   [[2, 1, 1]],
                   [[1, 2, 1]]])

    ys = np.array([[[2, 2, 3]],
                   [[1, 5, 1]],
                   [[4, 2, 2]],
                   [[-1, -2, -1]]])
    theta = 0.8

    out = predictor.predict(w, U, xs, ys, theta, triRel=True)
    expectedOut = np.array([True, False])
    assert (out == expectedOut).all()
