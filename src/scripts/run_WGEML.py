import numpy as np
import sys
from scipy.io import loadmat, savemat
from .. import dataPath
from ..data_preparation import save_and_load, prep_cross_valid, properly_formatted_inputs
from ..WGEML import create_values

# Usage : python3 -m src.scripts.run_WGEML [dataset] [relationship_2_char] [restricted]
# Ex. python3 -m src.scripts.run_WGEML KinFaceW-I fd unrestricted
# The above runs WGEML for the dataset KinFaceW-I with the father-daughter relationship
# And with the image-unrestricted setting in which case negative pairs are given to WGEML

# TODO: For the dimension of U, could it be 100 and PCA only drops the feature vectors to 200 
#       since we have in the paper "we use PCA to project each feature representation to a 
#       200-dimensional space and then set the reduced dimension as 100" The reduced dimension might be d
#       and not a truncation thing.

dataset = sys.argv[1]
relationship = sys.argv[2]
restricted = sys.argv[3]

# mat_to_cross_folds
pathToMat = dataPath + dataset + "/meta_data/" + relationship + "_pairs.mat"

# Unpickle the face descriptors
listOfFDs = save_and_load.unpickle_face_descriptors(dataset)

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
            U, w = create_values.get_all_values_for_a_relationship(posPairSet, negPairSet=negPairSet)
            ws.append(w)
            transformation_matrices.append(U)
        
        else:
            U, w = create_values.get_all_values_for_a_relationship(posPairSet)
            ws.append(w)
            transformation_matrices.append(U)

    # Once done with each training set, call save_w_and_U
    save_and_load.save_w_and_U(ws, transformation_matrices, relationship, dataset, restricted=restricted)

else:
    training, testing = prep_cross_valid.TSK_get_splits(pathToMat, True, relationship)
    ws = []
    transformation_matrices = []
    for i in range(len(training)):
        posPairSet, _ = properly_formatted_inputs.get_input_to_WGEML(training[i], None, listOfFDs)
        U, w = create_values.get_all_values_for_a_relationship(posPairSet)
        ws.append(w)
        transformation_matrices.append(U)
    save_and_load.save_w_and_U(ws, transformation_matrices, relationship, dataset)
    savemat(dataPath + "TSKinFace/splits/" + relationship + "_splits.mat", {"splits" : [training, testing]})
    