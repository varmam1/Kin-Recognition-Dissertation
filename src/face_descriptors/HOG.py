import numpy as np
from math import sqrt
import cv2

# Create a 2880-dim vector by having 9 dim vector for
# 16x16 blocks then for 8x8 blocks


def compute_gradients(img):
    """
    Given a colored image, computes the unsigned gradients for each point and
    returns the magnitude and angle for each point.

    Keyword Arguments:
    - img (np.array): An np array that represents the colored image

    Returns:
    - A tuple of the magnitude and angle np arrays.
        - magnitude (np.array): An array representing the magnitude of the
                                gradient at each pixel

        - angle (np.array): An array representing the angle of the gradient
                            with respect to the x axis at each pixel. Ranges
                            from 0 to 180.
    """
    img = np.float32(img)
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)
    magnitude, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)

    # If the image is colored, take max for each pixel along the 3 channels
    if len(img.shape) == 3:
        magnitude = np.amax(magnitude, axis=(2))
        angle = np.amax(angle, axis=(2))

    angle = angle % 180
    return (magnitude, angle)


def create_block_HOG_vector(blockMagnitudes, blockAngles):
    """
    Given the magnitudes and angles of the gradients of a block of an image,
    returns the 9-dimensional histogram for the block. The counts for each
    angle are weighted by the magnitude and by how far the angle is from the
    necessary angles. For example, if a pixel had a gradient with angle 30 and
    magnitude 2, it would contribute 1 to the 20 bin and 1 to the 40 bin.

    Keyword Arguments:
    - blockMagnitudes (np.array) : A numpy array of shape (w, h) which has the
                                   magnitudes of the gradient for each pixel in
                                   the block.

    - blockAngles (np.array) : A numpy array of shape (w, h) which has the
                               unsigned angles of the gradient for each pixel
                               in the block.

    Returns:
    - A numpy array of shape (9, ) which is the histogram of the given block
      as defined above.
    """
    vec = np.zeros(9)

    # For each cell, divide angle by 20 to find out cell it's in
    bins = blockAngles/20

    # Create the amount they weigh towards the lower bin and the upper bin.
    # If a value is an integer, its fully under weightsTowardsLower. So if a
    # value of an element is 2, then the corresponding values are 1 and 0.

    weightsTowardsLower = (20*(np.floor(bins)+1)-blockAngles)/20

    addToLowerBins = blockMagnitudes*weightsTowardsLower
    addToUpperBins = blockMagnitudes - addToLowerBins

    lowerBins = np.uint8(np.floor(bins))
    upperBins = np.uint8(np.ceil(bins) % 9)

    np.add.at(vec, lowerBins, addToLowerBins)
    np.add.at(vec, upperBins, addToUpperBins)

    return vec


def L2_norm_normalization(vec):
    """
    Returns the L2 norm of the given vector. This is defined as:

    v / sqrt(|v|^2 + epsilon^2)

    in the paper "Histograms of Oriented Gradients for Human Detection"
    where epsilon is a small regularization constant.

    Keyword Arguments:
    - vec (np.array) : A numpy array of the feature descriptor needed to be
                       normalized

    Returns:
    - The L2 norm of the vector as an np.array.
    """
    norm = np.linalg.norm(vec)
    epsilon = 0.01
    return vec/sqrt(norm**2 + epsilon ** 2)


def get_HOG_feature_vector(img):
    """
    Given an image, returns the HOG feature vector. If the image is 64x64,
    then the resulting vector will be 2880-dimensional.

    Keyword Arguments:
    - img (np.array) : An np array representation of the image

    Returns:
    - A Histogram of Gradients feature vector of the image.
    """
    magnitudes, angles = compute_gradients(img)
    vec = np.array([])

    # First get vectors for the 16x16 blocks

    firstSize = (int(img.shape[0]/16), int(img.shape[1]/16))
    for i in range(0, 16):
        for j in range(0, 16):
            blockMags = magnitudes[firstSize[0]*i: firstSize[0] * (i + 1),
                                   firstSize[1]*j: firstSize[1] * (j + 1)]
            blockAngles = angles[firstSize[0]*i: firstSize[0] * (i + 1),
                                 firstSize[1]*j: firstSize[1] * (j + 1)]
            blockVec = create_block_HOG_vector(blockMags, blockAngles)
            vec = np.append(vec, blockVec)

    # Then get the vectors for the 8x8 blocks

    secondSize = (int(img.shape[0]/8), int(img.shape[1]/8))
    secondBlockVecs = np.zeros((8, 8, 9))
    for i in range(0, 8):
        for j in range(0, 8):
            blockMags = magnitudes[secondSize[0]*i: secondSize[0] * (i + 1),
                                   secondSize[1]*j: secondSize[1] * (j + 1)]
            blockAngles = angles[secondSize[0]*i: secondSize[0] * (i + 1),
                                 secondSize[1]*j: secondSize[1] * (j + 1)]
            blockVec = create_block_HOG_vector(blockMags, blockAngles)
            vec = np.append(vec, blockVec)

    # TODO: As a potential improvement to the paper, normalize the HOG feature
    #       vectors as this improves performance and the paper doesn't seem to
    #       do this normalization. They just say "A 9-dimensional feature
    #       vector is computed for each block. Finally, we concatenate all
    #       these vectors together and get a 2880-dimensional feature vector"

    # TODO: For normalization of 8x8 blocks, use sliding window of 2 blocks by
    #       2 blocks and use the L2 norm as made above.

    return vec
