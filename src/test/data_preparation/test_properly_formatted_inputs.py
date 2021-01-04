import pytest
import numpy as np
from ...data_preparation import properly_formatted_inputs as pfi


# =================== Testing Getting WGEML Input Function ===================

def test_get_WGEML_inputs():
    posPairs = np.array([["test1", "test2"],
                         ["test3", "test4"],
                         ["test5", "test6"]])

    negPairs = np.array([["test1", "test3"],
                         ["test2", "test4"],
                         ["test3", "test6"]])

    fd1_map = {"test1" : np.array([0, 0, 0, 0]),
               "test2" : np.array([1, 1, 1, 1]),
               "test3" : np.array([2, 2, 2, 2]),
               "test4" : np.array([3, 3, 3, 3]),
               "test5" : np.array([4, 4, 4, 4]),
               "test6" : np.array([5, 5, 5, 5]),
               "notIn" : np.array([9, 9, 9, 9])}

    fd2_map = {"test1" : -np.array([0, 0]),
               "test2" : -np.array([1, 1]),
               "test3" : -np.array([2, 2]),
               "test4" : -np.array([3, 3]),
               "test5" : -np.array([4, 4]),
               "test6" : -np.array([5, 5]),
               "notIn" : -np.array([9, 9])}
    
    posX1 = np.array([np.array([0, 0, 0, 0]), 
                      np.array([2, 2, 2, 2]),
                      np.array([4, 4, 4, 4])])

    posY1 = np.array([np.array([1, 1, 1, 1]), 
                      np.array([3, 3, 3, 3]),
                      np.array([5, 5, 5, 5])])

    posX2 = np.array([-np.array([0, 0]), 
                      -np.array([2, 2]),
                      -np.array([4, 4])])

    posY2 = np.array([-np.array([1, 1]), 
                      -np.array([3, 3]),
                      -np.array([5, 5])])

    negX1 = np.array([np.array([0, 0, 0, 0]), 
                      np.array([1, 1, 1, 1]),
                      np.array([2, 2, 2, 2])])

    negY1 = np.array([2*np.ones(4), 
                      3*np.ones(4),
                      5*np.ones(4)])

    negX2 = np.array([-np.array([0, 0]), 
                      -np.array([1, 1]),
                      -np.array([2, 2])])

    negY2 = np.array([-np.array([2, 2]), 
                      -np.array([3, 3]),
                      -np.array([5, 5])])

    out = pfi.get_input_to_WGEML(posPairs, negPairs, [fd1_map, fd2_map])

    allSame = True
    allSame = allSame and (out[0][0][0] == posX1).all()
    allSame = allSame and (out[0][0][1] == posY1).all()
    
    allSame = allSame and (out[0][1][0] == posX2).all()
    allSame = allSame and (out[0][1][1] == posY2).all()

    allSame = allSame and (out[1][0][0] == negX1).all()
    allSame = allSame and (out[1][0][1] == negY1).all()

    allSame = allSame and (out[1][1][0] == negX2).all()
    allSame = allSame and (out[1][1][1] == negY2).all()

    assert allSame
