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
    np.testing.assert_equal(
        SIFT.get_octaves_and_blurring(img), expectedOctaves)


def test_getting_octaves_with_4_by_4_image_and_2_octaves_and_2_per_octave_with_high_blurring():
    img = np.float32(np.array([[2, 2, 2, 2],
                               [2, 2, 2, 2],
                               [3, 3, 3, 3],
                               [3, 3, 3, 3]]))
    out = SIFT.get_octaves_and_blurring(
        img, num_octaves=2, num_blurs=2, sigma=10)
    expectedOctaves = {}
    expectedOctaves[1] = np.array([img, np.ones((4, 4))*2.5])
    expectedOctaves[2] = np.array([[[2, 2], [3, 3]], np.ones((2, 2))*2.5])
    assert all(np.allclose(out[key], expectedOctaves[key]) for key in out)

# ===================== Testing Difference of Gaussians ======================


def test_difference_of_gaussians_on_3_images():
    octave = np.array([[[2, 2, 2, 2],
                        [2, 2, 2, 2],
                        [3, 3, 3, 3],
                        [3, 3, 3, 3]],

                       [[2.2, 2.2, 2.2, 2.2],
                        [2.2, 2.2, 2.2, 2.2],
                        [2.8, 2.8, 2.8, 2.8],
                        [2.8, 2.8, 2.8, 2.8]],

                       np.ones((4, 4)) * 2.5])

    expectedDoG = np.array([[[0.2,  0.2,  0.2,  0.2],
                             [0.2,  0.2,  0.2,  0.2],
                             [-0.2, -0.2, -0.2, -0.2],
                             [-0.2, -0.2, -0.2, -0.2]],

                            [[0.3,  0.3,  0.3,  0.3],
                             [0.3,  0.3,  0.3,  0.3],
                             [-0.3, -0.3, -0.3, -0.3],
                             [-0.3, -0.3, -0.3, -0.3]]])

    DoG = SIFT.difference_of_gaussians(octave)
    # Need to use isclose due to floating point
    assert np.isclose(DoG, expectedDoG).sum() == (
        expectedDoG.shape[0]*expectedDoG.shape[1]*expectedDoG.shape[2])


def test_difference_of_gaussians_on_3_black_images():
    octave = np.zeros((3, 64, 64))
    expectedOctave = np.zeros((2, 64, 64))
    assert np.array_equal(SIFT.difference_of_gaussians(octave), expectedOctave)

# ======================= Testing Getting the Hessian ========================


def test_getting_hessian_of_constant_array():
    dog = np.ones((3, 64, 64))
    expectedOut = np.zeros((3, 64, 64, 3, 3))
    assert np.array_equal(SIFT.get_hessian(dog), expectedOut)


def test_getting_hessian_of_array_():
    dog = np.fromfunction(lambda x, y, z: z**2, (3, 10, 10), dtype=int)
    expectedOut = np.zeros((3, 10, 10, 3, 3))
    for i in range(3):
        for j in range(10):
            for k in range(10):
                hess = np.zeros((3, 3))
                if k - 2 < 0 or k + 2 >= 10:
                    hess[2, 2] = 1
                else:
                    hess[2, 2] = 2
                expectedOut[i, j, k] = hess
    assert np.isclose(SIFT.get_hessian(dog), expectedOut).all()

# ================= Testing Getting Local Maxima and Minima ==================


def test_getting_maxima_and_minima_when_input_has_none():
    DoG = np.zeros((4, 64, 64))
    DoG[2, 2, 2] = 10
    DoG[2, 3, 2] = 10
    DoG[1, 1, 1] = -10
    DoG[2, 1, 1] = -10
    expectedOut = (np.zeros((0, 3)), np.zeros((0, 3)))
    assert np.array_equal(SIFT.get_max_and_min_of_DoG(DoG), expectedOut)


def test_getting_maxima_and_minima_when_input_has_one_of_each():
    DoG = np.zeros((4, 64, 64))
    DoG[2, 2, 2] = 10
    DoG[1, 1, 1] = -10
    expectedOut = (np.array([[1, 1, 1]]), np.array([[2, 2, 2]]))
    assert np.array_equal(SIFT.get_max_and_min_of_DoG(DoG), expectedOut)


def test_getting_maxima_and_minima_when_input_has_multiple_of_each():
    DoG = np.zeros((4, 64, 64))
    DoG[2, 2, 2] = 10
    DoG[2, 7, 8] = 100
    DoG[1, 1, 1] = -10
    DoG[1, 4, 6] = -100
    expectedOut = (np.array([[1, 1, 1], [1, 4, 6]]),
                   np.array([[2, 2, 2], [2, 7, 8]]))
    assert np.array_equal(SIFT.get_max_and_min_of_DoG(DoG), expectedOut)


# ===================== Testing Subpixel Extrema Function ====================


def test_getting_subpixel_extrema_when_extrema_on_pixel():
    diff_of_gauss = np.zeros((3, 10, 10))
    diff_of_gauss[1, 5, 5] = 100
    for i in range(-1, 2):
        for j in range(-1, 2):
            for k in range(-1, 2):
                if i != 0 or j != 0 or k != 0:
                    diff_of_gauss[i + 1, j + 5, k + 5] = 50
    out = SIFT.find_subpixel_maxima_and_minima(
        diff_of_gauss, np.array([[1, 5, 5]]))
    expectedOut = np.array([[1, 5, 5]])
    assert np.array_equal(out, expectedOut)


# ================= Testing Deleting Edge Responses Function =================


def test_deleting_edge_responses_when_no_edges():
    diff_of_gauss = np.zeros((3, 10, 10))
    diff_of_gauss[1, 5, 5] = 100
    for i in range(-1, 2):
        for j in range(-1, 2):
            for k in range(-1, 2):
                if i != 0 or j != 0 or k != 0:
                    diff_of_gauss[i + 1, j + 5, k + 5] = 50
    out = SIFT.remove_edge_responses(diff_of_gauss, np.array([[1, 5, 5]]))
    expectedOut = np.array([[1, 5, 5]])
    assert np.array_equal(out, expectedOut)


# ================== Testing Getting Orientations Function ===================


def test_getting_orientations_of_keypoint_with_90_orientation():
    DoG = np.zeros((3, 3, 3))
    DoG[1] = np.array([[0, -1, 0],
                       [0,  0, 0],
                       [0,  1, 0]])
    keypoint = np.array([[1, 1, 1]])
    expectedOut = {0: np.array([90])}
    assert expectedOut == SIFT.get_orientations_for_keypoint(DoG, keypoint)


def test_getting_orientations_of_keypoint_with_more_than_one_orientation():
    DoG = np.zeros((3, 3, 3))
    DoG[1] = np.array([[ 1, -1, 0],
                       [ 0,  0, 0],
                       [-1,  1, 0]])
    keypoint = np.array([[1, 1, 1]])
    expectedOutArr = np.array([90, 270])
    assert np.array_equal(expectedOutArr, SIFT.get_orientations_for_keypoint(DoG, keypoint)[0])
