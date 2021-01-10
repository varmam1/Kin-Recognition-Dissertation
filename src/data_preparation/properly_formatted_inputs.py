import numpy as np


def get_input_to_WGEML(posPairs, negPairs, face_descriptors):
    """
    Given the face descriptors, initializes the object that will be used to
    train the WGEML algorithm for one fold and for one kinship relationship.

    Keyword Arguments:
    - posPairs: A numpy array of lists in which the elements in the list have the
        kinship relationship described.

    - negPairs: The same as posPairs but doesn't have the kinship relationship.

    - face_descriptors: A list of the dictionaries which hold the path to the
        image as the key and the corresponding face descriptor. In our uses,
        will be the list [LBP_map, HOG_map, SIFT_map, VGG_map] although this
        can be changed easily.

    Returns:
    - The input into the WGEML algorithm with the necessary formatting.
    """
    posPairSet = []
    negPairSet = []
    for face_descriptor_map in face_descriptors:
        posDescriptors = np.array([[face_descriptor_map[k] for k in l] for l in posPairs])
        posX = posDescriptors[:, 0]
        posY = posDescriptors[:, 1]
        posViewTuple = (posX, posY)
        posPairSet.append(posViewTuple)
        
        if negPairs != None:
            negDescriptors = np.array([[face_descriptor_map[k] for k in l] for l in negPairs])
            negX = negDescriptors[:, 0]
            negY = negDescriptors[:, 1]
            negViewTuple = (negX, negY)
            negPairSet.append(negViewTuple)

    return (posPairSet, negPairSet)
