import numpy as np
from .. import dataPath
import pickle
from ..data_preparation.prep_cross_valid import NUMBER_OF_FOLDS

pathToDatasetSplits = dataPath + "TSKinFace/splits/"
num_FMD_pairs = 274
num_FMS_pairs = 285
num_FMSD_pairs = 228
num_pairs_in_test_set = 101
relationships = ["FS", "FD", "MS", "MD", "FMS", "FMD"]

for rel in relationships:

    splits = []
    training_splits = []
    child = rel[-1]
    num_non_fmsd = 0
    highest_val = 0

    # Get the proper number of values to be from each dataset proportionally
    if child == "S":
        num_non_fmsd = int(num_pairs_in_test_set * (num_FMS_pairs / (num_FMS_pairs + num_FMSD_pairs)))
        highest_val = num_FMS_pairs
    else:
        num_non_fmsd = int(num_pairs_in_test_set * (num_FMD_pairs / (num_FMD_pairs + num_FMSD_pairs)))
        highest_val = num_FMD_pairs
    num_fmsd = num_pairs_in_test_set - num_non_fmsd

    # Create the negative pairs where the pairs are just pairs of non-equal numbers essentially
    for fold in range(NUMBER_OF_FOLDS):
        test_set = []
        # Do it for the non-FMSD set
        for _ in range(num_non_fmsd):
            x = np.random.randint(low=1, high=highest_val)
            y = np.random.randint(low=1, high=highest_val - 1)
            if y == x:
                y = highest_val
            pair = []
            pair.append("FM" + child + "-" + str(x) + "-" + rel[0] + ".jpg")
            if len(rel) == 3:
                pair.append("FM" + child + "-" + str(x) + "-" + rel[1] + ".jpg")
            pair.append("FM" + child + "-" + str(y) + "-" + rel[-1] + ".jpg")
            test_set.append(pair)

        # Do it for the FMSD set
        for _ in range(num_fmsd):
            x = np.random.randint(low=1, high=num_FMSD_pairs)
            y = np.random.randint(low=1, high=num_FMSD_pairs - 1)
            if y == x:
                y = num_FMSD_pairs
            pair = []
            pair.append("FMSD" + "-" + str(x) + "-" + rel[0] + ".jpg")
            if len(rel) == 3:
                pair.append("FMSD" + "-" + str(x) + "-" + rel[1] + ".jpg")
            pair.append("FMSD" + "-" + str(y) + "-" + rel[-1] + ".jpg")
            test_set.append(pair)

        test_set = np.array(test_set)
        splits.append(test_set)

    # Create training sets for each test set here=
    tupledSplits = tuple(splits)
    for i in range(NUMBER_OF_FOLDS):
        training = tupledSplits[:i] + tupledSplits[i+1:]
        training_splits.append(np.concatenate(training))
    
    if len(rel) == 3:
        for i in range(NUMBER_OF_FOLDS):
            splits[i] = np.concatenate((splits[i][:,[0,2]], splits[i][:,[1,2]]))
            training_splits[i] = np.concatenate((training_splits[i][:,[0,2]], training_splits[i][:,[1,2]]))

    toSave = open(pathToDatasetSplits + "neg_" + rel.lower() + "_splits.pkl", "wb")
    pickle.dump({"training": training_splits, "testing" : splits}, toSave)
    toSave.close()
