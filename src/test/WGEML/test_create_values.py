import pytest
import numpy as np
from ...WGEML import create_values


# ======================= Testing Getting S_p and D_p ========================


def test_getting_diff_matrix():
    x_view = np.array([[1, 2], [2, 3], [3, 4]])
    y_view = np.array([[0, 0], [1, 1], [2, 2]])
    expectedOut = np.array([[1, 2], [2, 4]])
    assert (create_values.get_graphs(x_view, y_view) == expectedOut).all()


# ====================== Testing Getting D_1p and D_2p =======================


def test_getting_D_1p_and_D_2p_with_two_neighbors_and_2_dims():
    x_view = np.array([[1, 2], [0, 0], [3, 4], [1, 3], [5, 7]])
    y_view = np.array([[1, 4], [2, 3], [6, 5], [3, 4], [2, 1]])
    x_neighbors = np.array([[0, 3], [0, 1], [2, 3], [0, 3], [2, 4]])
    y_neighbors = np.array([[0, 1], [0, 1], [2, 3], [1, 3], [1, 4]])
    expected_D_1p = np.array([[3.8, 4.6], [4.6, 8.4]])
    expected_D_2p = np.array([[5.7, 4.7], [4.7, 7.0]])
    out = create_values.get_penalty_graph_2(
        x_view, y_view, x_neighbors, y_neighbors)
    D_1_correct = np.isclose(out[0], expected_D_1p).all()
    D_2_correct = np.isclose(out[1], expected_D_2p).all()
    assert D_1_correct and D_2_correct


# ==================== Testing Top d Eigenvectors function ===================


def test_getting_proper_top_2_eigenvectors():
    eigenvalues = np.zeros((3, 3))
    eigenvalues[0, 0] = 4
    eigenvalues[1, 1] = 2
    eigenvalues[2, 2] = 3
    eigenvectors = np.array([[1, 0, 1],
                             [1, 2, 1],
                             [1, 3, 2]])
    A = np.dot(np.dot(eigenvectors, eigenvalues), np.linalg.inv(eigenvectors))
    out = create_values.get_top_d_eigenvectors(A, np.identity(3), 2)
    expectedOut = np.array(
        [np.transpose(eigenvectors)[0], np.transpose(eigenvectors)[2]])
    expectedOut = expectedOut/(expectedOut.sum(axis=1)[:, None])
    assert np.isclose(expectedOut, out/(out.sum(axis=1)[:, None])).all()


# ================= Testing main U_p and w_p getter function =================


def test_getting_U_and_w_for_a_basic_input_with_one_descriptor_and_dimension_10():
    x_pos = np.array([np.array([i, 0, 0, 0]) for i in range(1, 11)])
    x_neg = np.array([np.array([0, 0, i, 0]) for i in range(1, 11)])
    y_pos = np.array([np.array([0, i, 0, 0]) for i in range(1, 11)])
    y_neg = np.array([np.array([0, 0, 0, i]) for i in range(1, 11)])
    posPairSet = ((x_pos, y_pos), )
    negPairSet = ((x_neg, y_neg), )
    out = create_values.get_all_values_for_a_relationship(
        posPairSet, negPairSet, 2)
    expected_U_0 = np.zeros((4, 2))
    expected_U_0[2][0] = 0.70710678
    expected_U_0[3][0] = -0.70710678
    expected_U_0[0][1] = -0.8928498
    expected_U_0[1][1] = -0.45035456
    assert np.isclose(expected_U_0, out[0][0]).all() and (
        out[1] == np.array([1.0])).all()
