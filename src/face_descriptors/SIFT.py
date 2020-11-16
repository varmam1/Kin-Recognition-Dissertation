import numpy as np
import cv2
from math import sqrt

# Creates a 6272-dimensional feature vector for the image
# This is done by splitting up the image into 16x16 cells and then using a
# sliding window of 2 cells by 2 cells to create 7x7 patches with which you
# get a 128 dimensional vector from each patch.


def get_octaves_and_blurring(img, num_octaves=4, num_blurs=5, sigma=sqrt(2)):
    """
    Given a grayscaled image and the number of octaves and how many images
    should be in each octave, returns a dictionary of the octaves. The value
    with key, i, element of the list corresponds to the i'th octave and the
    j'th element in the octave is the original image blurred j times.

    Keyword Arguments:
    - img (np.array) : A grayscaled image that the octaves need to be created
                       from.
    - num_octaves (int) : The number of octaves that need to be made.
    - num_blurs (int) : The number of images should be in each octave.

    Returns:
    - A dictionary from int to the octave which is a numpy list of images.
    """
    octaves = {}

    size = img.shape
    for i in range(num_octaves):
        octave = np.zeros((num_blurs, size[0], size[1]))
        octave[0] = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        for j in range(1, num_blurs):
            octave[j] = cv2.GaussianBlur(octave[j - 1], (0, 0),
                                         sigmaX=sigma**j, sigmaY=sigma**j)
        size = (int(size[0]/2), int(size[1]/2))
        octaves[i + 1] = octave

    return octaves


def difference_of_gaussians(octave):
    """
    Given an octave which is a list of the images ordered from no blur to most
    blur, returns the difference in the blurred images.

    Keyword Arguments:
    - octave (np.array) : The octave which has the images in order from no blur
                          to most blurred

    Returns:
    - A numpy array which is the Difference of Gaussians which approximates
      the Laplacian of Gaussian.
    """
    DoG = np.zeros((octave.shape[0] - 1, octave.shape[1], octave.shape[2]))
    for i in range(1, octave.shape[0]):
        DoG[i - 1] = octave[i] - octave[i - 1]
    return DoG
