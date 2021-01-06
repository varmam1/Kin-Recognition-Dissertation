import numpy as np
import cv2
import sys
import os
import pickle
from ..face_descriptors import VGG, SIFT, LBP, HOG
from .. import dataPath

dataset = sys.argv[1]
pathToDataset = dataPath + dataset

LBP_map = {}
HOG_map = {}
SIFT_map = {}

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
            LBP_map[img] = LBP.create_LBP_feature_vector(image)
            HOG_map[img] = HOG.get_HOG_feature_vector(image)
            SIFT_map[img] = SIFT.paper_main_function_SIFT(image)

VGG_map = dict(zip(pathsToAllImages, VGG.get_VGG_face_descriptor(images)))

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
