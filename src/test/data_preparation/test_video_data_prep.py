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
