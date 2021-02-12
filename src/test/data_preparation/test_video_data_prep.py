import pytest
import numpy as np
from ...data_preparation import video_data_prep

def test_get_random_frames():
    np.random.seed(0)
    img = np.array([[[1, 2, 3], [2, 3, 4]], 
           [[0, 0, 0], [1, 1, 1]],
           [[5, 2, 2], [2, 3, 2]],
           [[6, 2, 3], [6, 2, 4]]])
    out = video_data_prep.get_random_frames(img, amnt=2)
    expectedOut = img[2:4]
    assert (out == expectedOut).all()


def test_get_fds_from_set_of_frames():
    # For the sake of simplicity, doing a scalar to vector fn
    img = np.arange(10)
    def fd(scalar):
        return scalar*np.ones(5)
    expectedOut = np.ones(5)*4.5
    out = video_data_prep.get_specified_face_descriptor(img, fd)
    assert np.isclose(out, expectedOut).all()
