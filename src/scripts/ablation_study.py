import numpy as np
from ..prediction import predictor
from ..data_preparation import save_and_load, prep_cross_valid
from .. import dataPath
import pickle
import sys
import os
import csv

# Usage : python3 -m src.scripts.ablation_study [dataset] [restricted]
# This script will run every combination of face descriptors and get the
# accuracies for each relationship and output it into a CSV file.

THETA = 0.6

dataset = sys.argv[1]
restricted = sys.argv[2]
relationships = ["fs", "fd", "ms", "md"]
if dataset.lower() == "tskinface":
    relationships.append("fms")
    relationships.append("fmd")

if restricted.lower() != "unrestricted" and restricted.lower() != "restricted":
    restricted = None

# A list ordered by ["HOG", "LBP", "SIFT", "VGG"] and the corresponding value
# in this list checks whether it is included or not
fd_names = np.array(["HOG", "LBP", "SIFT", "VGG"])
out = open('out/' + dataset + ('_' + restricted if restricted is not None else "") + '.csv', 'w', newline='')
csv_out = csv.writer(out)
csv_out.writerow(['FDs Used'] + relationships)

for bit_fds_included in range(1, 16):
    # face_descriptors_included = [True, True, True, True]
    face_descriptors_included = np.array([bool(bit_fds_included & (1<<n)) for n in range(4)])
    print(str(fd_names[face_descriptors_included]))

    # Load in face descriptors

    pathToDataset = dataPath + dataset + "/"
    all_fd_maps = []
    for i, fd_map_name in enumerate(os.listdir(pathToDataset + "fds")):
        all_fd_maps.append(pickle.load(open(pathToDataset + "fds/" + fd_map_name, "rb")))

    rel_accs = [str(fd_names[face_descriptors_included])]
    for relationship in relationships:
        w, U = save_and_load.load_w_and_U(dataset, relationship, restricted)

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
            predictionsPos = predictor.predict_if_kin_1(w[i], U[i], xs, ys, THETA, triRel=(len(relationship) == 3), fds_included=face_descriptors_included)

            neg = negSets[i]

            negXS = np.array([[fd[img] for fd in all_fd_maps] for img in neg[:, 0]])
            negYS = np.array([[fd[img] for fd in all_fd_maps] for img in neg[:, 1]])

            predictionsNeg = predictor.predict_if_kin_1(w[i], U[i], negXS, negYS, THETA, triRel=(len(relationship) == 3), fds_included=face_descriptors_included)
            
            # Check how many are correct 
            acc = (predictionsPos.sum() + len(predictionsNeg) - predictionsNeg.sum())/(len(predictionsPos) + len(predictionsNeg))

            # Save the accuracy in a list
            accuracies.append(acc)

        accuracies = np.array(accuracies)

        # Output the list of accuracies and the mean of it
        print(dataset + "-" + relationship + "-" + ("" if restricted is None else restricted) + ": " + str(accuracies.mean()))
        rel_accs.append(accuracies.mean())

    csv_out.writerow(rel_accs)

out.close()
