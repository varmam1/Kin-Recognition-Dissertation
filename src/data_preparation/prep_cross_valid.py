import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import KFold


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


def get_all_training_splits(splits):
    """
    Given the splits, returns the list of the training splits. One fold is
    used for testing and the rest of the folds are used for training. This
    will return a list of the combined splits where the corresponding testing
    split is the corresponding fold in the input. For example, the first set
    of pairs in the output will be the training set and the first set of pairs
    in the input is the test fold.

    Keyword Arguments:
    - splits: A list of numpy arrays with shape (N, 2) which represent the
    pairs of either positive or negative relationship.

    Returns:
    - A list of numpy arrays of shape ((NUM_FOLDS - 1)*N, 2) which represent
    the training splits.
    """
    tupledSplits = tuple(splits)
    trainingSplits = []
    for i in range(len(tupledSplits)):
        training = tupledSplits[:i] + tupledSplits[i+1:]
        trainingSplits.append(np.concatenate(training))
    
    return trainingSplits


# ===================== TSKinFace Dataset Prep Functions =====================


def TSK_get_splits(path_to_mat_file, shuffle_data):
    """

    Keyword Arguments:
    - path_to_mat_file (str): A string which is just the path to the
        mat file that has the pairs for TSKinFace

    - shuffle_data (bool) : Whether or not to shuffle the pairs around before
        assigning them to splits.

    Returns:
    - A list of numpy arrays of shape ((NUM_FOLDS - 1)*N, 2) which represent
    the training splits and a list of numpy arrays which represents the test
    splits.
    """
    data = loadmat(path_to_mat_file,
        matlab_compatible=False, struct_as_record=False)
    pairs = np.char.strip(data["pairs"])

    splitter = KFold(n_splits=NUMBER_OF_FOLDS, shuffle=shuffle_data)

    training = []
    testing = []
    for train, test in splitter.split(pairs):
        training.append(pairs[train])
        testing.append(pairs[test])
    return (training, testing)
    