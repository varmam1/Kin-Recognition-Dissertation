import numpy as np
from .. import dataPath
import pickle

pathToDatasetSplits = dataPath + "TSKinFace/splits/"
num_FMD_pairs = 274
num_FMS_pairs = 285
num_FMSD_pairs = 228
num_pairs_in_test_set = 101
relationships = ["FS", "FD", "MS", "MD", "FMS", "FMD"]

for rel in relationships:
    # Want to pick tests proportionally from fms/d and fmsd
    # Want to also pick 5 different test sets for each relationship
    # Since need to do a diff one for each fold
    # Want to create around 102 pairs (whether that is a two or tri rel)

    neg_test_sets = []
    child = rel[-1]
    num_non_fmsd = 0
    highest_val = 0
    if child == "S":
        num_non_fmsd = int(num_pairs_in_test_set * (num_FMS_pairs / (num_FMS_pairs + num_FMSD_pairs)))
        highest_val = num_FMS_pairs
    else:
        num_non_fmsd = int(num_pairs_in_test_set * (num_FMD_pairs / (num_FMD_pairs + num_FMSD_pairs)))
        highest_val = num_FMD_pairs
    num_fmsd = num_pairs_in_test_set - num_non_fmsd

    for fold in range(5):
        test_set = []
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

        for _ in range(num_fmsd):
            x = np.random.randint(low=1, high=num_FMSD_pairs)
            y = np.random.randint(low=1, high=num_FMSD_pairs - 1)
            if y == x:
                y = highest_val
            pair = []
            pair.append("FMSD" + "-" + str(x) + "-" + rel[0] + ".jpg")
            if len(rel) == 3:
                pair.append("FMSD" + "-" + str(x) + "-" + rel[1] + ".jpg")
            pair.append("FMSD" + "-" + str(y) + "-" + rel[-1] + ".jpg")
            test_set.append(pair)

        neg_test_sets.append(np.array(test_set))
    toSave = open(pathToDatasetSplits + "neg_" + rel.lower() + "_splits.pkl", "wb")
    pickle.dump(neg_test_sets, toSave)
    toSave.close()
