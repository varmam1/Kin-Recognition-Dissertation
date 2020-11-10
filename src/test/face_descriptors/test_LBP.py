import pytest
import numpy as np
from ...face_descriptors import LBP

# =================== Testing Neighborhood Function ===================

def test_get_val_of_neighborhood_if_all_equal():
    neighborhood = np.array([[0, 0, 0],
                             [0, 0, 0],
                             [0, 0, 0]])
    assert LBP.get_number_from_neighborhood(neighborhood) == 0

def test_get_val_of_neighborhood_when_all_above_except_to_the_left():
    neighborhood = np.array([[2, 2, 2],
                             [0, 1, 2],
                             [2, 2, 2]])
    assert LBP.get_number_from_neighborhood(neighborhood) == 127

# ======================= Testing LBP Function ========================
