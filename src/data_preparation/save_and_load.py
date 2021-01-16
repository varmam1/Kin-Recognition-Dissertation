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
        fd_maps.append(pickle.load(open(pathToFDs + f, "rb")))
    return fd_maps


def save_w_and_U(ws, trans_matrices, relationship, dataset, restricted=None):
    """
    For a relationship, ex. father-daughter, and the dataset, save all of the
    w and U vectors and matrices for each training set in a map and saves on
    disk.

    Keyword Arguments:
    - ws: A list of the w vector for each training set of the cross validation
    - trans_matrices: A list of the U matrix for each training set
    - relationship: What relationship is being tested (ex. father-daughter, etc.)
    - dataset: The name of the dataset that this is being done with. Ex. KinFaceW-I
    - restricted: A string setting for WGEML about whether it's restricted or not.
    """
    pathToSave = dataPath + dataset + "/WGEML_out/"
    if restricted is not None:
        pathToSave = pathToSave + restricted + "/"
    if not os.path.exists(pathToSave):
        os.makedirs(pathToSave)
    WGEML_results = open(pathToSave + relationship + "_out.pkl", "wb")
    pickle.dump({"w": ws, "U" : trans_matrices}, WGEML_results)
    WGEML_results.close()


def load_w_and_U(dataset, relationship, restricted=None):
    """
    Loads the w and U values from disk given the necessary configuration.

    Keyword Arguments:
    - dataset: The name of the dataset that this is being done with. Ex. KinFaceW-I
    - relationship: What relationship is being tested (ex. father-daughter, etc.)
    - restricted: A string setting for WGEML about whether it's restricted or not.

    Returns:
    - A pair, (w, U).
    """
    pathToSave = dataPath + dataset + "/WGEML_out/"
    if restricted is not None:
        pathToSave = pathToSave + restricted + "/"
    WGEML_results = open(pathToSave + relationship + "_out.pkl", "rb")
    data = pickle.load(WGEML_results)
    return (data['w'], data['U'])
