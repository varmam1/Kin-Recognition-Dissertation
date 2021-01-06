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


def save_w_and_U(ws, trans_matrices, relationship, dataset):
    """
    For a relationship, ex. father-daughter, and the dataset, save all of the
    w and U vectors and matrices for each training set in a map and saves on
    disk.

    Keyword Arguments:
    - ws: A list of the w vector for each training set of the cross validation
    - trans_matrices: A list of the U matrix for each training set
    - relationship: What relationship is being tested (ex. father-daughter, etc.)
    - dataset: The name of the dataset that this is being done with. Ex. KinFaceW-I
    """
    pathToSave = dataPath + dataset + "/WGEML_out/"
    WGEML_results = open(pathToSave + "/" + relationship + "_out.pkl", "wb")
    pickle.dump({"w": ws, "U" : trans_matrices}, WGEML_results)
    WGEML_results.close()
