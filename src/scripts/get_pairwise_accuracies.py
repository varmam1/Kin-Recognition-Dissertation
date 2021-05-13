import numpy as np
from ..prediction import predictor
from ..data_preparation import save_and_load, prep_cross_valid
from .. import dataPath
from . import DATASETS
import pickle
import sys
import os
import csv

THETA = 0.6
trainDataset = sys.argv[1]
restricted = sys.argv[2]
pathToTrainDataset = dataPath + trainDataset + "/"
relationships = ["fs", "fd", "ms", "md"]

if restricted.lower() != "unrestricted" and restricted.lower() != "restricted":
    restricted = None

out = open('out/pairwise_accs/' + trainDataset + ('_' + restricted if restricted is not None else "") + '.csv', 'w', newline='')
csv_out = csv.writer(out)
csv_out.writerow(['Test Dataset Used'] + relationships)

# For each dataset there is, use that for the testing set
for testDataset in DATASETS:
    
    print(testDataset)

    allRelationshipAccs = []
    
    # Do this for each relationship
    for relationship in relationships:
        # Load in face descriptors
        w, U = save_and_load.load_w_and_U(trainDataset, relationship, restricted)

        pathToTestDataset = dataPath + testDataset + "/"
        all_fd_maps = []
        for fd_map_name in os.listdir(pathToTestDataset + "fds"):
            all_fd_maps.append(pickle.load(open(pathToTestDataset + "fds/" + fd_map_name, "rb")))

        # Then load in the test sets

        testSets = []
        negSets = []

        if testDataset.lower() == "tskinface":
            testSets = pickle.load(open(dataPath + "TSKinFace/splits/" + relationship + "_splits.pkl", "rb"))["testSets"]
            negSets = pickle.load(open(dataPath + "TSKinFace/splits/neg_" + relationship + "_splits.pkl", "rb"))["testing"]
        else:
            testSets, negSets = prep_cross_valid.get_splits_for_positive_and_negative_pairs(pathToTestDataset + "meta_data/" + relationship + "_pairs.mat")

        accuracies = []
        # For each fold
        for i in range(len(w)):
            # use the fds and create the xs and ys pairs for the predictor
            test = testSets[i]
            xs = np.array([[fd[img] for fd in all_fd_maps] for img in test[:, 0]])
            ys = np.array([[fd[img] for fd in all_fd_maps] for img in test[:, 1]])
            
            # Run the prediction algo
            predictionsPos = predictor.predict(w[i], U[i], xs, ys, THETA, triRel=(len(relationship) == 3))

            neg = negSets[i]

            negXS = np.array([[fd[img] for fd in all_fd_maps] for img in neg[:, 0]])
            negYS = np.array([[fd[img] for fd in all_fd_maps] for img in neg[:, 1]])

            predictionsNeg = predictor.predict(w[i], U[i], negXS, negYS, THETA, triRel=(len(relationship) == 3))
            
            # Check how many are correct 
            acc = (predictionsPos.sum() + len(predictionsNeg) - predictionsNeg.sum())/(len(predictionsPos) + len(predictionsNeg))

            # Save the accuracy in a list
            accuracies.append(acc)

        allRelationshipAccs.append(np.round(np.array(accuracies).mean(), 4))
    csv_out.writerow([testDataset] + allRelationshipAccs)

csv_out.writerow([])
out.close()
