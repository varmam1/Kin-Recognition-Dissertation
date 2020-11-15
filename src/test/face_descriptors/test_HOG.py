import pytest
import numpy as np
from math import sqrt
from ...face_descriptors import HOG

# =================== Testing Compute Gradient Function ===================

def test_compute_gradient_grayscale():
    img = np.array([[3, 2, 1],
                    [3, 2, 1],
                    [3, 2, 1]])

    expectedMag = np.array([[0, 2, 0],
                            [0, 2, 0],
                            [0, 2, 0]])

    expectedAngle = np.array([[0, 0, 0],
                              [0, 0, 0],
                              [0, 0, 0]])

    out = HOG.compute_gradients(img)
    # Since magnitudes are floats here, require isclose to test if theyre similar
    # And check that every element in the array is similar enough
    assert (np.isclose(out[0], expectedMag).sum() == img.shape[0]*img.shape[1]) and np.array_equal(out[1], expectedAngle)

def test_compute_gradient_colored():
    # Check that the gradient works as necessary for colored images too. 
    img = np.array([[[1, 2, 3], [2, 2, 2], [3, 2, 1]],
                    [[1, 2, 3], [2, 2, 2], [3, 2, 1]],
                    [[1, 2, 3], [2, 2, 2], [3, 2, 1]]])
    
    expectedMag = np.array([[0, 2, 0],
                            [0, 2, 0],
                            [0, 2, 0]])
    
    expectedAngle = np.array([[0, 0, 0],
                              [0, 0, 0],
                              [0, 0, 0]])
    
    out = HOG.compute_gradients(img)

    assert (np.isclose(out[0], expectedMag).sum() == img.shape[0]*img.shape[1]) and np.array_equal(out[1], expectedAngle)

# =================== Testing Block Vector Creator Function ===================

def test_block_histogram_creator():
    mags = np.array([[1,  2, 4],
                     [6,  5, 8],
                     [2, 10, 3]])

    angles = np.array([[ 0, 20,  75],
                       [30, 40, 125],
                       [25, 35, 100]])

    out = HOG.create_block_HOG_vector(mags, angles)
    # [0, 20, 40, 60, 80, 100, 120, 140, 160]
    expectedOut = np.array([1.0, 9.0, 16.0, 1.0, 3.0, 3.0, 6.0, 2.0, 0.0])
    assert np.array_equal(out, expectedOut)

def test_block_histogram_creator_on_empty_input():
    assert np.array_equal(np.zeros(9), HOG.create_block_HOG_vector(np.array([]), np.array([])))

# ====================== Testing L2 Norm Normalization Function =====================

def test_L2_norm_on_zero_vector():
    vec = np.zeros(9)
    assert np.array_equal(HOG.L2_norm_normalization(vec), vec)

def test_L2_norm_on_standard_vector():
    vec = np.array([3, 4])
    expectedNorm = np.array([3/sqrt(25 + 0.0001), 4/sqrt(25 + 0.0001)])
    assert np.array_equal(HOG.L2_norm_normalization(vec), expectedNorm)

# =================== Testing Overall HOG Feature Vector Function ===================

def test_HOG_vector_on_black_image():
    image = np.zeros((64, 64, 3))
    expectedOut = np.zeros(2880)
    assert np.array_equal(expectedOut, HOG.get_HOG_feature_vector(image))
