import pytest
import numpy as np
from ...face_descriptors import LBP

# =================== Testing Uniform Vals Function ===================

uniform_values = np.array([ 0,  1,   2,   3,   4,   6,   7,   8,
        12,  14,  15, 16,  24,  28,  30,  31,  32,  48,  56,
        60,  62,  63, 64,  96, 112, 120, 124, 126, 127, 128,
        129, 131, 135, 143, 159, 191, 192, 193, 195, 199, 207,
        223, 224, 225, 227, 231, 239, 240, 241, 243, 247, 248,
        249, 251, 252, 253, 254, 255])

def test_generate_uniform_values_returns_all_uniform_values_from_0_to_255():
    assert np.array_equal(LBP.generate_uniform_values(), uniform_values)

# =================== Testing Neighborhood Function ===================

def test_get_val_of_neighborhood_if_all_equal():
    neighborhood = np.array([[0, 0, 0],
                             [0, 0, 0],
                             [0, 0, 0]])
    assert LBP.get_number_from_neighborhood(neighborhood, uniform_values) == 0

def test_get_val_of_neighborhood_when_all_above_except_to_the_left():
    neighborhood = np.array([[2, 2, 2],
                             [0, 1, 2],
                             [2, 2, 2]])
    assert LBP.get_number_from_neighborhood(neighborhood, uniform_values) == 127

def test_get_val_of_neighborhood_when_LBP_value_is_non_uniform():
    neighborhood = np.array([[2, 0, 2],
                             [0, 1, 0],
                             [2, 0, 2]])
    assert LBP.get_number_from_neighborhood(neighborhood, uniform_values) == -1

# ======================= Testing LBP_image Function ========================

def test_LBP_image_on_black_image():
    image = np.array([[0, 0, 0, 0],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0]])

    expected_output = np.array([[0, 0, 0, 0],
                                [0, 0, 0, 0],
                                [0, 0, 0, 0],
                                [0, 0, 0, 0]])
    assert np.array_equal(LBP.LBP_image(image, uniform_values), expected_output)

def test_LBP_image_on_empty_image_returns_no_LBPs():
    image = np.array([])
    assert np.array_equal(LBP.LBP_image(image, uniform_values), image)

def test_LBP_image_on_normal_image():
    image = np.array([[ 1, 50, 108],
                      [39, 25,  28],
                      [40, 48, 189]])

    expected_output = np.array([[56,   8,  0],
                                [-1, 254, -1],
                                [ 8,   8,  0]])

    assert np.array_equal(LBP.LBP_image(image, uniform_values), expected_output)
                    
# ======================= Testing LBP Vector Function ========================

def test_LBP_vector_on_black_image():
    image = np.zeros((64, 64, 3)).astype(np.uint8)
    expected_output = np.zeros(59*64)
    for i in range(0, 64):
        expected_output[59*i + 1] = 64
    assert np.array_equal(LBP.create_LBP_feature_vector(image), expected_output)
