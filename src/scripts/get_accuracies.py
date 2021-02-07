import numpy as np
from ..prediction import predictor
from ..data_preparation import save_and_load, prep_cross_valid
from .. import dataPath
import pickle
import sys
import os

# Usage : python3 -m src.scripts.get_accuracies [dataset] [relationship_2_char] [restricted]
# This will get the accuracies for the dataset, relationship, restricted
# configuration. 

THETA = 0.6

dataset = sys.argv[1]
relationship = sys.argv[2]
restricted = sys.argv[3]

if restricted.lower() != "unrestricted" and restricted.lower() != "restricted":
    restricted = None

w, U = save_and_load.load_w_and_U(dataset, relationship, restricted)

# Load in face descriptors

pathToDataset = dataPath + dataset + "/"
all_fd_maps = []
for fd_map_name in os.listdir(pathToDataset + "fds"):
    all_fd_maps.append(pickle.load(open(pathToDataset + "fds/" + fd_map_name, "rb")))

# Then load in the test sets

testSets = []
negSets = []

if dataset.lower() == "tskinface":
    testSets = pickle.load(open(dataPath + "TSKinFace/splits/" + relationship + "_splits.pkl", "rb"))["testSets"]
    negSets = pickle.load(open(dataPath + "TSKinFace/splits/neg_" + relationship + "_splits.pkl", "rb"))["testing"]
else:
    testSets, negSets = prep_cross_valid.get_splits_for_positive_and_negative_pairs(pathToDataset + "meta_data/" + relationship + "_pairs.mat")

accuracies = []
# For each fold
for i in range(len(w)):
    # use the fds and create the xs and ys pairs for the predictor
    test = testSets[i]
    xs = np.array([[fd[img] for fd in all_fd_maps] for img in test[:, 0]])
    ys = np.array([[fd[img] for fd in all_fd_maps] for img in test[:, 1]])
    
    # Run the prediction algo
    predictionsPos = predictor.predict_if_kin_1(w[i], U[i], xs, ys, THETA, triRel=(len(relationship) == 3))

    neg = negSets[i]

    negXS = np.array([[fd[img] for fd in all_fd_maps] for img in neg[:, 0]])
    negYS = np.array([[fd[img] for fd in all_fd_maps] for img in neg[:, 1]])

    predictionsNeg = predictor.predict_if_kin_1(w[i], U[i], negXS, negYS, THETA, triRel=(len(relationship) == 3))
    
    # Check how many are correct 
    acc = (predictionsPos.sum() + len(predictionsNeg) - predictionsNeg.sum())/(len(predictionsPos) + len(predictionsNeg))

    # Save the accuracy in a list
    accuracies.append(acc)

accuracies = np.array(accuracies)

# Output the list of accuracies and the mean of it
print(dataset + "-" + relationship + "-" + ("" if restricted is None else restricted) + ": " + str(accuracies))
print(dataset + "-" + relationship + "-" + ("" if restricted is None else restricted) + ": " + str(accuracies.mean()))
