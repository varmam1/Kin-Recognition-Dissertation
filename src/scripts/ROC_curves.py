import numpy as np
from ..prediction import predictor
from ..data_preparation import save_and_load, prep_cross_valid
from .. import dataPath
from sklearn.metrics import roc_curve, auc
from scipy import interp
import pickle
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# Aim to put each dataset's ROC curve on the same graph

datasets = ["KinFaceW-I", "KinFaceW-II", "TSKinFace"]

# Load in face descriptors

relationships = ["fs", "fd", "ms", "md"]

plt.figure(figsize=(8,5))

for dataset in datasets:
    pathToDataset = dataPath + dataset + "/"
    all_fd_maps = save_and_load.unpickle_face_descriptors(dataset)

    restricted = ["unrestricted", "restricted"]
    if dataset == "TSKinFace":
        restricted = [None]
        relationships = ["fs", "fd", "ms", "md", 'fms', 'fmd']

    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    all_tpr = []

    for setting in restricted:
        for relationship in relationships:
            w, U = save_and_load.load_w_and_U(dataset, relationship, setting)

            # Then load in the test sets

            testSets = []
            negSets = []

            if dataset.lower() == "tskinface":
                testSets = pickle.load(open(dataPath + "TSKinFace/splits/" + relationship + "_splits.pkl", "rb"))["testSets"]
                negSets = pickle.load(open(dataPath + "TSKinFace/splits/neg_" + relationship + "_splits.pkl", "rb"))["testing"]
            else:
                testSets, negSets = prep_cross_valid.get_splits_for_positive_and_negative_pairs(pathToDataset + "meta_data/" + relationship + "_pairs.mat")

            # For each fold
            for i in range(len(w)):
                # use the fds and create the xs and ys pairs for the predictor
                test = testSets[i]
                xs = np.array([[fd[img] for fd in all_fd_maps] for img in test[:, 0]])
                ys = np.array([[fd[img] for fd in all_fd_maps] for img in test[:, 1]])
                
                # Run the prediction algo
                predictionsPosScore = predictor.get_similarity(w[i], U[i], xs, ys, triRel=(len(relationship) == 3))

                neg = negSets[i]

                negXS = np.array([[fd[img] for fd in all_fd_maps] for img in neg[:, 0]])
                negYS = np.array([[fd[img] for fd in all_fd_maps] for img in neg[:, 1]])

                predictionsNegScore = predictor.get_similarity(w[i], U[i], negXS, negYS, triRel=(len(relationship) == 3))

                yTest = np.concatenate((np.ones(len(test)), np.zeros(len(neg))))
                if len(relationship) == 3:
                    yTest = np.concatenate((np.ones(len(test)//2), np.zeros(len(neg)//2)))
                fpr, tpr, _ = roc_curve(yTest, np.concatenate((predictionsPosScore, predictionsNegScore)))
                mean_tpr += interp(mean_fpr, fpr, tpr)
                mean_tpr[0] = 0.0
                roc_auc = auc(fpr, tpr)

    numDiv = 4*5*2
    if dataset == "TSKinFace":
        numDiv = 6*5
    mean_tpr /= numDiv
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr,
            label='Mean ROC ' + dataset + ' (area = %0.2f)' % mean_auc, lw=2)

plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Each Dataset Averaged Over Fold, Relationship and Setting')
plt.legend(loc="lower right")
plt.savefig("out/ROC")

# Aim To average over relationships as well
plt.figure(figsize=(8,5))
relationships = ["fs", "fd", "ms", "md", 'fms', 'fmd']

for relationship in relationships:
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    all_tpr = []
    if len(relationship) == 3:
        datasets = ["TSKinFace"]
    
    for dataset in datasets:
        pathToDataset = dataPath + dataset + "/"
        all_fd_maps = save_and_load.unpickle_face_descriptors(dataset)

        restricted = ["unrestricted", "restricted"]
        if dataset == "TSKinFace":
            restricted = [None]

        for setting in restricted:
            w, U = save_and_load.load_w_and_U(dataset, relationship, setting)

            # Then load in the test sets

            testSets = []
            negSets = []

            if dataset.lower() == "tskinface":
                testSets = pickle.load(open(dataPath + "TSKinFace/splits/" + relationship + "_splits.pkl", "rb"))["testSets"]
                negSets = pickle.load(open(dataPath + "TSKinFace/splits/neg_" + relationship + "_splits.pkl", "rb"))["testing"]
            else:
                testSets, negSets = prep_cross_valid.get_splits_for_positive_and_negative_pairs(pathToDataset + "meta_data/" + relationship + "_pairs.mat")

            # For each fold
            for i in range(len(w)):
                # use the fds and create the xs and ys pairs for the predictor
                test = testSets[i]
                xs = np.array([[fd[img] for fd in all_fd_maps] for img in test[:, 0]])
                ys = np.array([[fd[img] for fd in all_fd_maps] for img in test[:, 1]])
                
                # Run the prediction algo
                predictionsPosScore = predictor.get_similarity(w[i], U[i], xs, ys, triRel=(len(relationship) == 3))

                neg = negSets[i]

                negXS = np.array([[fd[img] for fd in all_fd_maps] for img in neg[:, 0]])
                negYS = np.array([[fd[img] for fd in all_fd_maps] for img in neg[:, 1]])

                predictionsNegScore = predictor.get_similarity(w[i], U[i], negXS, negYS, triRel=(len(relationship) == 3))

                yTest = np.concatenate((np.ones(len(test)), np.zeros(len(neg))))
                if len(relationship) == 3:
                    yTest = np.concatenate((np.ones(len(test)//2), np.zeros(len(neg)//2)))
                fpr, tpr, _ = roc_curve(yTest, np.concatenate((predictionsPosScore, predictionsNegScore)))
                mean_tpr += interp(mean_fpr, fpr, tpr)
                mean_tpr[0] = 0.0
                roc_auc = auc(fpr, tpr)

    numDiv = 5*2*2 + 5
    if len(relationship) == 3:
        numDiv = 5
    mean_tpr /= numDiv
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr,
            label='Mean ROC ' + relationship.upper() + ' (area = %0.2f)' % mean_auc, lw=2)

plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Each Relationship Averaged Over Fold, Dataset and Setting')
plt.legend(loc="lower right")
plt.savefig("out/ROC_rel")
