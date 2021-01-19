import numpy as np
import cv2
import sys
import os
import pickle
from ..data_preparation import PCA
from ..face_descriptors import VGG, SIFT, LBP, HOG
from .. import dataPath

dataset = sys.argv[1]
pathToDataset = dataPath + dataset

LBP_fds = []
HOG_fds = []
SIFT_fds = []

pathsToAllImages = []
images = []

for relationship in os.listdir(pathToDataset + "/images/"):
    path = pathToDataset + "/images/" + relationship + "/"
    for img in os.listdir(path):
        pathToImage = path + img
        if pathToImage[-3:] == "png" or pathToImage[-3:] == "jpg":
            pathsToAllImages.append(img)
            image = cv2.imread(pathToImage)
            images.append(image)
            LBP_fds.append(LBP.create_LBP_feature_vector(image))
            HOG_fds.append(HOG.get_HOG_feature_vector(image))
            SIFT_fds.append(SIFT.paper_main_function_SIFT(image))

LBP_fds = np.array(LBP_fds)
HOG_fds = np.array(HOG_fds)
SIFT_fds = np.array(SIFT_fds)
VGG_fds = VGG.get_VGG_face_descriptor(np.array(images))


# Reduce the dimensions of the face descriptors
reduced_LBP_fds = PCA.reduce_dimensions(LBP_fds)
reduced_HOG_fds = PCA.reduce_dimensions(HOG_fds)
reduced_SIFT_fds = PCA.reduce_dimensions(SIFT_fds)
reduced_VGG_fds = PCA.reduce_dimensions(VGG_fds)

# Match the vectors to the image names
LBP_map = dict(zip(pathsToAllImages, reduced_LBP_fds))
HOG_map = dict(zip(pathsToAllImages, reduced_HOG_fds))
SIFT_map = dict(zip(pathsToAllImages, reduced_SIFT_fds))
VGG_map = dict(zip(pathsToAllImages, reduced_VGG_fds))

# Save the face descriptors on disk

LBP_face_descriptors_file = open(pathToDataset + "/fds/LBP_face_descriptors.pkl", "wb")
pickle.dump(LBP_map, LBP_face_descriptors_file)
LBP_face_descriptors_file.close()

HOG_face_descriptors_file = open(pathToDataset + "/fds/HOG_face_descriptors.pkl", "wb")
pickle.dump(HOG_map, HOG_face_descriptors_file)
HOG_face_descriptors_file.close()

SIFT_face_descriptors_file = open(pathToDataset + "/fds/SIFT_face_descriptors.pkl", "wb")
pickle.dump(SIFT_map, SIFT_face_descriptors_file)
SIFT_face_descriptors_file.close()

VGG_face_descriptors_file = open(pathToDataset + "/fds/VGG_face_descriptors.pkl", "wb")
pickle.dump(VGG_map, VGG_face_descriptors_file)
VGG_face_descriptors_file.close()
