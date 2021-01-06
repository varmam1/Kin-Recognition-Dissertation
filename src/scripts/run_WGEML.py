import numpy as np
import sys
from scipy.io import loadmat
from .. import dataPath
from ..data_preparation import save_and_load, prep_cross_valid, properly_formatted_inputs
from ..WGEML import create_values

# Usage : python3 -m src.scripts.run_WGEML [dataset] [relationship_2_char] [restricted]
# Ex. python3 -m src.scripts.run_WGEML KinFaceW-I fd unrestricted
# The above runs WGEML for the dataset KinFaceW-I with the father-daughter relationship
# And with the image-unrestricted setting in which case negative pairs are given to WGEML

dataset = sys.argv[1]
relationship = sys.argv[2]
restricted = sys.argv[3]

# mat_to_cross_folds
# Check if restricted or not and if restricted, ignore the neg_splits
# Run the get_cv_config
# Unpickle the face descriptors
# For each training set, run the maps_w_pairs_to_input and then WGEML on that and add to a list both w and U
# Once done with each training set, call save_w_and_U

pathToMat = dataPath + dataset + "/meta_data/" + relationship + "_pairs.mat"
positiveSplits, negativeSplits = prep_cross_valid.get_splits_for_positive_and_negative_pairs(pathToMat)

posTrainingSplits = prep_cross_valid.get_all_training_splits(positiveSplits)
negTrainingSplits = prep_cross_valid.get_all_training_splits(negativeSplits)

listOfFDs = save_and_load.unpickle_face_descriptors(dataset)

ws = []
transformation_matrices = []
for i in range(len(posTrainingSplits)):
    posPairs = posTrainingSplits[i]
    negPairs = negTrainingSplits[i]

    posPairSet, negPairSet = properly_formatted_inputs.get_input_to_WGEML(posPairs, negPairs, listOfFDs)

    if restricted.lower() == "unrestricted":
        U, w = create_values.get_all_values_for_a_relationship(posPairSet, negPairSet, 10, False)
        ws.append(w)
        transformation_matrices.append(U)
    
    else:
        U, w = create_values.get_all_values_for_a_relationship(posPairSet, [], 10, True)
        ws.append(w)
        transformation_matrices.append(U)

save_and_load.save_w_and_U(ws, transformation_matrices, relationship, dataset)
