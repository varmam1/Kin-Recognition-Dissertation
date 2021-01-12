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


# ===================== Testing Getting Training Splits ======================


def test_get_training_splits():
    fold1 = np.arange(4).reshape((2, 2))
    fold2 = np.arange(4, 10).reshape((3, 2))
    fold3 = np.arange(10, 12).reshape((1, 2))
    fold4 = np.arange(12, 16).reshape((2, 2))

    splits = [fold1, fold2, fold3, fold4]

    out = prep_cross_valid.get_all_training_splits(splits)
    expectedTrain1 = np.arange(4, 16).reshape((6, 2))
    expectedTrain2 = np.array([[ 0,  1],
                               [ 2,  3],
                               [10, 11],
                               [12, 13],
                               [14, 15]])
    expectedTrain3 = np.array([[ 0,  1],
                               [ 2,  3],
                               [ 4,  5],
                               [ 6,  7],
                               [ 8,  9],
                               [12, 13],
                               [14, 15]])
    expectedTrain4 = np.arange(12).reshape((6, 2))

    assert (out[0] == expectedTrain1).all() and (out[1] == expectedTrain2).all() and (out[2] == expectedTrain3).all() and (out[3] == expectedTrain4).all()


# =============== Testing Getting Training Sets For TSKinFace ================


def test_getting_training_splits_for_TSK_dataset():
    out = prep_cross_valid.TSK_get_splits("src/test/data_preparation/testTSK.mat", False, "fd")
    origInput = np.arange(10, 30).reshape(10, 2).astype(str)
    expectedOutTrain = [origInput[2:], origInput[np.r_[0:2, 4:10]], origInput[np.r_[0:4, 6:10]], origInput[np.r_[0:6, 8:10]], origInput[:8]]
    expectedOutTest = [origInput[:2], origInput[2:4], origInput[4:6], origInput[6:8], origInput[8:]]

    trainSame = True
    testSame = True
    for i in range(len(out[0])):
        trainSame = trainSame and (out[0][i] == expectedOutTrain[i]).all()
        testSame = testSame and (out[1][i] == expectedOutTest[i]).all()
    
    assert trainSame and testSame


def test_getting_training_splits_for_TSK_dataset_with_tri_relationship():
    out = prep_cross_valid.TSK_get_splits("src/test/data_preparation/fdm_test.mat", False, "fdm")
    origInput = np.arange(10, 25).reshape(5, 3).astype(str)

    expectedOutTrain = []
    expectedOutTrain.append(np.array([['13', '15'],
                                        ['16', '18'],
                                        ['19', '21'],
                                        ['22', '24'],
                                        ['14', '15'],
                                        ['17', '18'],
                                        ['20', '21'],
                                        ['23', '24']]))

    expectedOutTrain.append(np.array([['10', '12'],
                                        ['16', '18'],
                                        ['19', '21'],
                                        ['22', '24'],
                                        ['11', '12'],
                                        ['17', '18'],
                                        ['20', '21'],
                                        ['23', '24']]))

    expectedOutTrain.append(np.array([['10', '12'],
                                        ['13', '15'],
                                        ['19', '21'],
                                        ['22', '24'],
                                        ['11', '12'],
                                        ['13', '15'],
                                        ['20', '21'],
                                        ['23', '24']]))

    expectedOutTrain.append(np.array([['10', '12'],
                                        ['13', '15'],
                                        ['16', '18'],
                                        ['22', '24'],
                                        ['11', '12'],
                                        ['13', '15'],
                                        ['17', '18'],
                                        ['23', '24']]))

    expectedOutTrain.append(np.array([['10', '12'],
                                        ['13', '15'],
                                        ['16', '18'],
                                        ['19', '21'],
                                        ['11', '12'],
                                        ['13', '15'],
                                        ['17', '18'],
                                        ['20', '21']]))

    expectedOutTest = [np.array([['10', '12'], ['11', '12']]),
                       np.array([['13', '15'], ['14', '15']]),
                       np.array([['16', '18'], ['17', '18']]),
                       np.array([['19', '21'], ['20', '21']]),
                       np.array([['22', '24'], ['23', '24']])]

    trainSame = True
    testSame = True
    for i in range(len(out[0])):
        trainSame = trainSame and (out[0][i] == expectedOutTrain[i]).all()
        testSame = testSame and (out[1][i] == expectedOutTest[i]).all()
    
    assert testSame
