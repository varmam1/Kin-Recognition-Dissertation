import numpy as np
import cv2
import sys
import os
import pickle
from ..face_descriptors import VGG, SIFT, LBP, HOG

dataset = sys.argv[1]
pathToDataset = "data/" + dataset

LBP_map = {}
HOG_map = {}
SIFT_map = {}
VGG_map = {}

for relationship in os.listdir(pathToDataset + "/images/"):
    path = pathToDataset + "/images/" + relationship + "/"
    for img in os.listdir(path):
        pathToImage = path + img
        if pathToImage[-3:] == "png" or pathToImage[-3:] == "jpg":
            image = cv2.imread(pathToImage)
            LBP_map[pathToImage] = LBP.create_LBP_feature_vector(image)
            HOG_map[pathToImage] = HOG.get_HOG_feature_vector(image)
            SIFT_map[pathToImage] = SIFT.paper_main_function_SIFT(image)
            # TODO: Potentially change the VGG one to do it all at once
            # VGG_map[pathToImage] = VGG.get_VGG_face_descriptor(image[np.newaxis])[0]
        else:
            print(pathToImage)

# Save the face descriptors on disk

LBP_face_descriptors_file = open(pathToDataset + "/LBP_face_descriptors.pkl", "wb")
pickle.dump(LBP_map, LBP_face_descriptors_file)
LBP_face_descriptors_file.close()

HOG_face_descriptors_file = open(pathToDataset + "/HOG_face_descriptors.pkl", "wb")
pickle.dump(HOG_map, HOG_face_descriptors_file)
HOG_face_descriptors_file.close()

SIFT_face_descriptors_file = open(pathToDataset + "/SIFT_face_descriptors.pkl", "wb")
pickle.dump(SIFT_map, SIFT_face_descriptors_file)
SIFT_face_descriptors_file.close()

# VGG_face_descriptors_file = open(pathToDataset + "VGG_face_descriptors.pkl", "wb")
# pickle.dump(VGG_map, VGG_face_descriptors_file)
# VGG_face_descriptors_file.close()
