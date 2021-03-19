import numpy as np
import sys
import pickle
from scipy.io import loadmat, savemat
from .. import dataPath
from ..data_preparation import save_and_load, prep_cross_valid, properly_formatted_inputs
from ..WGEML import create_values

# Usage : python3 -m src.scripts.run_WGEML [dataset] [relationship_2_char] [restricted] [optional: exclude list]
# Ex. python3 -m src.scripts.run_WGEML KinFaceW-I fd unrestricted
# The above runs WGEML for the dataset KinFaceW-I with the father-daughter relationship
# And with the image-unrestricted setting in which case negative pairs are given to WGEML

dim_of_trans_matrix = 10

dataset = sys.argv[1]
relationship = sys.argv[2]
restricted = sys.argv[3]
exclusions = []
if len(sys.argv) == 5:
    exclusions = sys.argv[4].split(",")

# mat_to_cross_folds
pathToMat = dataPath + dataset + "/meta_data/" + relationship + "_pairs.mat"

# Unpickle the face descriptors
listOfFDs = save_and_load.unpickle_face_descriptors(dataset, exclude=exclusions)

if dataset != "TSKinFace":
    positiveSplits, negativeSplits = prep_cross_valid.get_splits_for_positive_and_negative_pairs(pathToMat)

    # Run the get_cv_config
    posTrainingSplits = prep_cross_valid.get_all_training_splits(positiveSplits)
    negTrainingSplits = prep_cross_valid.get_all_training_splits(negativeSplits)

    # For each training set, run the maps_w_pairs_to_input and then run 
    # WGEML on that and add to a list both w and U
    ws = []
    transformation_matrices = []
    for i in range(len(posTrainingSplits)):
        posPairs = posTrainingSplits[i]
        negPairs = negTrainingSplits[i]

        posPairSet, negPairSet = properly_formatted_inputs.get_input_to_WGEML(posPairs, negPairs, listOfFDs)

        if restricted.lower() == "unrestricted":
            U, w = create_values.get_all_values_for_a_relationship(posPairSet, negPairSet=negPairSet, dim_of_U=dim_of_trans_matrix)
            ws.append(w)
            transformation_matrices.append(U)
        
        else:
            U, w = create_values.get_all_values_for_a_relationship(posPairSet, dim_of_U=dim_of_trans_matrix)
            ws.append(w)
            transformation_matrices.append(U)

    # Once done with each training set, call save_w_and_U
    save_and_load.save_w_and_U(ws, transformation_matrices, relationship, dataset, restricted=restricted)

else:
    training, testing = prep_cross_valid.TSK_get_splits(pathToMat, True, relationship)
    training_neg = pickle.load(open(dataPath + dataset + "/splits/neg_" + relationship + "_splits.pkl", "rb"))["training"]
    ws = []
    transformation_matrices = []
    for i in range(len(training)):
        posPairSet, negPairSet = properly_formatted_inputs.get_input_to_WGEML(training[i], training_neg[i], listOfFDs)
        U, w = create_values.get_all_values_for_a_relationship(posPairSet, negPairSet=negPairSet, dim_of_U=dim_of_trans_matrix)
        ws.append(w)
        transformation_matrices.append(U)
    save_and_load.save_w_and_U(ws, transformation_matrices, relationship, dataset)
    trainTestSets = open(dataPath + "TSKinFace/splits/" + relationship + "_splits.pkl", "wb")
    pickle.dump({"trainSets": training, "testSets" : testing}, trainTestSets)
    trainTestSets.close()
    