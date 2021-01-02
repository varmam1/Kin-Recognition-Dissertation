import numpy as np
import pytest
from ...data_preparation import prep_cross_valid


# ======================= Testing Main Folds Function ========================


def test_main_function_to_get_folds_from_path():
    path = "src/test/data_preparation/test.mat"
    out = prep_cross_valid.get_splits_for_positive_and_negative_pairs(path)
    positiveFolds = np.array([[["test10", 'test11']], [["test30", 'test31']],
     [["test50", 'test51']], [["test70", 'test71']], [["test90", 'test91']]])
    negativeFolds = np.array([[["test00", 'test01']], [["test20", 'test21']],
     [["test40", 'test41']], [["test60", 'test61']], [["test80", 'test81']]])
    expectedOut = (positiveFolds, negativeFolds)
    assert (out[0] == expectedOut[0]).all() and (out[1] == expectedOut[1]).all()
