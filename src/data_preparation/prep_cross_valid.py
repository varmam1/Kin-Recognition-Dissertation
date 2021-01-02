import numpy as np
from scipy.io import loadmat


# This file is to prepare the splits and positive and negative data pairs
NUMBER_OF_FOLDS = 5


# ====================== KinFaceW preparation functions ======================


def get_positive_and_negative_pairs(data):
    """
    Given the data from the mat file from KinFaceW, returns a tuple
    of the positive pairs and the negative pairs.

    Keyword Arguments:
    - data: The data obtained from the KinFaceW mat file in that format.
    This should be a dict which has the key 'pairs' which has the information.

    Returns:
    - A tuple of (positivePairs, negativePairs)
    """
    return (data['pairs'][data['pairs'][:,1] == 1], data['pairs'][data['pairs'][:,1] == 0])


def get_splits(data_of_pairs):
    """
    Splits up the data into the given folds that are prepared in the original
    data. Returns only the names of the images.

    Keyword Argument:
    - data_of_pairs: The data to be split into the necessary folds utilizing
    the KinFaceW format.

    Returns:
    - A list of the folds in which each fold is only the list of pairs of names
    of the images.
    """
    pairsFolded = []

    for i in range(1, NUMBER_OF_FOLDS + 1):
        pairs = data_of_pairs[data_of_pairs[:, 0] == i][:, 2:]
        newPairFold = []
        for pair in pairs:
            newPairFold.append(np.array([pair[0][0], pair[1][0]]))
        pairsFolded.append(np.array(newPairFold))
    
    return pairsFolded


def get_splits_for_positive_and_negative_pairs(path_to_mat_file):
    """
    Given the path to the KinFaceW mat file for a relationship, returns
    a tuple of the positive and negative pairs split properly based on the
    folds.

    Keyword Arguments:
    - path_to_mat_file (str): A string which is just the path to the
        mat file obtained from KinFaceW

    Returns:
    - A pair of the list of positive and negative pairs split by fold.
    """
    data = loadmat(path_to_mat_file,
            matlab_compatible=False, struct_as_record=False)

    positive, negative = get_positive_and_negative_pairs(data)
    return (get_splits(positive), get_splits(negative))
