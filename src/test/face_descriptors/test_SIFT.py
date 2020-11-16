import pytest
import numpy as np
from ...face_descriptors import SIFT

# ========================= Testing Octaves and Blurs ========================

def test_getting_octaves_with_64_by_64_black_image():
    img = np.zeros((64, 64))
    expectedOctaves = {}
    expectedOctaves[1] = np.zeros((5, 64, 64))
    expectedOctaves[2] = np.zeros((5, 32, 32))
    expectedOctaves[3] = np.zeros((5, 16, 16))
    expectedOctaves[4] = np.zeros((5,  8,  8))
    np.testing.assert_equal(SIFT.get_octaves_and_blurring(img), expectedOctaves)

def test_getting_octaves_with_4_by_4_image_and_2_octaves_and_2_per_octave_with_high_blurring():
    img = np.float32(np.array([[2, 2, 2, 2],
                               [2, 2, 2, 2],
                               [3, 3, 3, 3],
                               [3, 3, 3, 3]]))
    out = SIFT.get_octaves_and_blurring(img, num_octaves=2, num_blurs=2, sigma=10)
    expectedOctaves = {}
    expectedOctaves[1] = np.array([img, np.ones((4, 4))*2.5])
    expectedOctaves[2] = np.array([[[2, 2], [3, 3]], np.ones((2, 2))*2.5])
    assert all(np.allclose(out[key], expectedOctaves[key]) for key in out)
