import numpy as np
import cv2
from scipy import ndimage as ndi
from math import sqrt

# Creates a 6272-dimensional feature vector for the image
# This is done by splitting up the image into 16x16 cells and then using a
# sliding window of 2 cells by 2 cells to create 7x7 patches with which you
# get a 128 dimensional vector from each patch.

#########################
# Scale Space Functions #
#########################


def get_octaves_and_blurring(img, num_octaves=4, num_blurs=5, sigma=1.6, k=sqrt(2)):
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
                                         sigmaX=sigma*(k**j),
                                         sigmaY=sigma*(k**j))
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

################################
# Keypoint Selection Functions #
################################


def get_hessian(diff_of_gauss):
    """
    Given a difference of Gaussians, returns the Hessian matrix for each point
    in the difference of Gaussians.

    Keyword Arguments:
    - diff_of_gauss (np.array): Difference of Gaussians of size (blurs, w, h)

    Returns:
    - An np.array of size (blurs, w, h, 3, 3) which is the Hessian Matrix at
    each element of the DoG.
    """
    first_order_gradient = np.gradient(diff_of_gauss)
    hessian = np.empty(diff_of_gauss.shape + (diff_of_gauss.ndim,
                                              diff_of_gauss.ndim), dtype=diff_of_gauss.dtype)
    for x, gradient_x in enumerate(first_order_gradient):
        second_grad_wrt_x = np.gradient(gradient_x)
        for y, grad_x_y in enumerate(second_grad_wrt_x):
            hessian[:, :, :, x, y] = grad_x_y
    return hessian


def get_max_and_min_of_DoG(DoG):
    """
    Given an octave's Difference of Gaussians, returns the coordinates of the
    local maxima and local minima in the Difference of Gaussians.

    Keyword Arguments:
    - DoG (np.array) : The Difference of Gaussians for one octave. This should
                       be a 3D numpy array.

    Returns:
    - A tuple (minCoords, maxCoords) in which minCoords is an array of size
    (n, 3) where n is the number of local minima and is a list of the
    coordinates of the local minima. maxCoords is similar to minCoords expect
    it holds the coordinates of the local maxima.
    """
    footprint = np.ones((3, 3, 3))
    footprint[1, 1, 1] = 0

    filteredMax = ndi.maximum_filter(DoG, footprint=footprint)
    filteredMin = ndi.minimum_filter(DoG, footprint=footprint)
    minCoords = np.asarray(np.where(DoG < filteredMin)).T
    maxCoords = np.asarray(np.where(DoG > filteredMax)).T
    return (minCoords, maxCoords)


def find_subpixel_maxima_and_minima(DoG, extrema):
    """
    Given the Difference of Gaussians and the local extrema in the DoG, finds
    the subpixel extrema to get a more accurate keypoint.

    Keyword Arguments:
    - DoG (np.array) : The difference of Gaussians for an octave.
    - extrema (np.array) : The coordinates in the DoG of the extrema.

    Returns:
    - A np array of the coordinates of the more accurate keypoints
    """ 
    subpixelExtrema = np.array([])
    hessian = get_hessian(DoG)
    gradients = np.gradient(DoG)
    for coordinate in extrema:
        hess = DoG[coordinate]
        grad_vector = np.array([])
        for grad_i in gradients:
            grad_vector = np.append(grad_vector, [grad_i[coordinate]])
        x_hat = - np.dot(np.linalg.inv(hess), grad_vector)
        subpixelExtrema = np.append(subpixelExtrema, x_hat)
    return subpixelExtrema


# TODO: Required to remove low contrast keypoints and then edge keypoints.
