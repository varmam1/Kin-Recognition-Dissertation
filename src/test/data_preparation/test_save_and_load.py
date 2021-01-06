import pytest
import numpy as np
import pickle
import shutil
from ...data_preparation import save_and_load
from ... import dataPath

def test_load_fds_from_disk():
    # Just to test that no errors occur
    try:
        save_and_load.unpickle_face_descriptors("KinFaceW-I")
    except:
        assert False
    assert True


def test_save_w_and_U():
    w1 = np.array([1, 2, 3])
    w2 = np.array([2, 3, 4])
    w = [w1, w2]

    U1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    U2 = -U1 - 2
    U = [U1, U2]

    save_and_load.save_w_and_U(w, U, "testRel", "test")

    # Load the files, check if it's what is wanted then delete the saved files
    out = pickle.load(open(dataPath + "test/WGEML_out/testRel_out.pkl", "rb"))
    shutil.rmtree(dataPath + "test")

    correct = (out["w"][0] == w1).all() and (out["w"][1] == w2).all()
    correct = correct and (out["U"][0] == U1).all() and (out["U"][1] == U2).all()
    assert correct
