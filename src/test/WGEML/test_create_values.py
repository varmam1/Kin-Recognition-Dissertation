import pytest
import numpy as np
from ...WGEML import create_values


# ======================= Testing Getting S_p and D_p ========================


def test_getting_diff_matrix():
    x_view = np.array([[1, 2], [2, 3], [3, 4]])
    y_view = np.array([[0, 0], [1, 1], [2, 2]])
    expectedOut = np.array([[1, 2], [2, 4]])
    assert (create_values.get_diff_mat(x_view, y_view) == expectedOut).all()


# ====================== Testing Getting D_1p and D_2p =======================


def test_getting_D_1p_and_D_2p_with_two_neighbors_and_2_dims():
    x_view = np.array([[1, 2], [0, 0], [3, 4], [1, 3], [5, 7]])
    y_view = np.array([[1, 4], [2, 3], [6, 5], [3, 4], [2, 1]])
    x_neighbors = np.array([[0, 3], [0, 1], [2, 3], [0, 3], [2, 4]])
    y_neighbors = np.array([[0, 1], [0, 1], [2, 3], [1, 3], [1, 4]])
    expected_D_1p = np.array([[3.8, 4.6], [4.6, 8.4]])
    expected_D_2p = np.array([[5.7, 4.7], [4.7, 7.0]])
    out = create_values.get_penalty_graphs(x_view, y_view, x_neighbors, y_neighbors)
    D_1_correct = np.isclose(out[0], expected_D_1p).all()
    D_2_correct = np.isclose(out[1], expected_D_2p).all()
    assert D_1_correct and D_2_correct
