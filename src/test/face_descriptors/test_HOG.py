import pytest
import numpy as np
from ...face_descriptors import HOG

# =================== Testing Compute Gradient Function ===================

def test_compute_gradient():
    img = np.array([[3, 2, 1], [3, 2, 1], [3, 2, 1]])
    expectedMag = np.array([[0, 2/255.0, 0],[0, 2/255.0, 0],[0, 2/255.0, 0]])
    expectedAngle = np.array([[0, 180, 0],[0, 180, 0],[0, 180, 0]])
    out = HOG.compute_gradients(img)
    # Since magnitudes are floats here, require isclose to test if theyre similar
    # And check that every element in the array is similar enough
    assert (np.isclose(out[0], expectedMag).sum() == img.shape[0]*img.shape[1]) and np.array_equal(out[1], expectedAngle)
