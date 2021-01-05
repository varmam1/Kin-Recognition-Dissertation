import numpy as np
import pickle
import os
from .. import dataPath


def unpickle_face_descriptors(dataset):
    """
    Given the dataset, gets the face descriptors already calculated and loads
    them in as a list of the maps.

    Keyword Argument:
    - dataset (str): A string which represents the dataset that the fds are
    obtained from. 

    Returns:
    - A list of the maps of the face descriptors for the images in the
    dataset.
    """
    pathToFDs = dataPath + dataset + "/fds/"
    fd_maps = []
    for f in os.listdir(pathToFDs):
        fd_maps.append(pickle.load(open(f, "rb")))
    return fd_maps


def save_w_and_U(ws, trans_matrices, relationship):
    """

    """