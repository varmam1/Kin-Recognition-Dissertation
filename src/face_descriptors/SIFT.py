import numpy as np
import cv2
from scipy import ndimage as ndi
from math import sqrt

# Creates a 6272-dimensional feature vector for the image
# This is done by splitting up the image into 16x16 cells and then using a
# sliding window of 2 cells by 2 cells to create 7x7 patches with which you
# get a 128 dimensional vector from each patch.

MAX_PIXEL_VALUE = 255
MAX_ITERATIONS = 1000

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
    - diff_of_gauss (np.array): Difference of Gaussians for an octave of size
                                (blurs, w, h)

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
    - A np array of the coordinates of the more accurate keypoints (floats)
    """
    subpixelExtrema = []
    hessian = get_hessian(DoG)
    gradients = np.gradient(DoG)
    for coordinate in extrema:
        new_coords = coordinate
        offset = np.zeros(DoG.ndim)
        outsideImage = False
        subpixel_keypoint = coordinate
        for dummy in range(MAX_ITERATIONS):
            # Calculate the subpixel keypoint via Taylor Expansion
            hess = hessian[new_coords[0], new_coords[1], new_coords[2]]
            grad_vector = np.array([])
            for grad_i in gradients:
                grad_vector = np.append(
                    grad_vector, [grad_i[new_coords[0], new_coords[1], new_coords[2]]])
            if np.array_equal(grad_vector, np.zeros(3)):
                break
            offset = - np.dot(np.linalg.inv(hess), grad_vector)
            if (offset - 0.5 <= 0).all():
                # If the offset is less than 0.5 in all dims then this is the
                # subpixel keypoint you want.
                subpixel_keypoint = new_coords + offset
                break
            new_coords = np.around(new_coords + offset).astype(int)

            # If the new coordinates are outside the DoG, give up.
            if (new_coords < 0).any() or (new_coords >= np.array(DoG.shape())).any():
                outsideImage = True
                break
            
        # If iterated more than the max amount of times, give up and move on.
        if dummy >= MAX_ITERATIONS or outsideImage:
            continue

        # Check what the value at subPixelCoord is is and if it's too small,
        # throw it out
        val_at_subpixel_coord = DoG[coordinate[0], coordinate[1],
                           coordinate[2]] + 0.5 * np.dot(grad_vector, subpixel_keypoint)

        if abs(val_at_subpixel_coord) >= 0.03 * MAX_PIXEL_VALUE:
            subpixelExtrema.append(subpixel_keypoint)

    return np.array(subpixelExtrema)


def remove_edge_responses(diff_of_gauss, keypoints):
    """
    Given the difference of Gaussians and the potential keypoints, deletes any
    keypoints that are edges.

    Keyword Arguments:
    - diff_of_gauss (np.array): The difference of Gaussians for an octave.
    - keypoints (np.array) : The coordinates in the DoG of the potential
                             keypoints

    Returns:
    - The coordinates of the keypoints that weren't edges.
    """
    hessian = get_hessian(diff_of_gauss)
    nonEdgeKeypoints = []
    for subCoords in keypoints:
        nearestIntKeypoint = np.around(subCoords).astype(int)
        # Only want the hessian for x and y, not sigma
        hess = hessian[nearestIntKeypoint[0],
                       nearestIntKeypoint[1], nearestIntKeypoint[2], 1:, 1:]
        determinant = np.linalg.det(hess)
        if determinant < 0:
            continue
        trace = np.trace(hess)
        ratio_of_eigenvalues = 10
        threshold = ((ratio_of_eigenvalues + 1)**2)/ratio_of_eigenvalues
        if (trace**2)/determinant < threshold:
            nonEdgeKeypoints.append(subCoords)
    return np.array(nonEdgeKeypoints)

#################################
# Keypoint Descriptor Functions #
#################################

def get_orientations_for_keypoint(diff_of_gauss, keypoints, sigma=1.6, k=sqrt(2)):
    """
    Given the difference of Gaussians and keypoints, will return the
    orientations for each keypoint as a map.

    Keyword Arguments:
    - diff_of_gauss: The difference of Gaussians
    - keypoints: A list of the coordinates of the keypoints.
    - sigma: The original sigma used to create the blurs
    - k: The original k used to create the blurs

    Returns:
    - A map (int -> np.array) of the index of the keypoint to their
      orientations. The index represents the index of the keypoint in the
      original array.
    """
    mapOfKeypointsToOrientation = {}
    for i in range(len(keypoints)):
        keypoint = keypoints[i]
        nearestIntKeypoint = np.around(keypoint).astype(int)
        scaledImg = diff_of_gauss[nearestIntKeypoint[0]]
        diffXKernel = np.array([[ 0, 0, 0],
                                [-1, 0, 1],
                                [ 0, 0, 0]])
        diffYKernel = np.transpose(diffXKernel)
        diffX = cv2.filter2D(scaledImg, -1, diffXKernel)
        diffY = cv2.filter2D(scaledImg, -1, diffYKernel)
        magnitude, angle = cv2.cartToPolar(diffX, diffY, angleInDegrees=True)
        # Assumption that keypoint isn't ever going to be on 0th blur level
        sizeOfCircularWindow = round(sigma * 1.5 * (k**nearestIntKeypoint[0]))
        anglesArea = angle[max(0, nearestIntKeypoint[1] - int(sizeOfCircularWindow/2)):min(diff_of_gauss.shape[1], nearestIntKeypoint[1] + int(sizeOfCircularWindow/2) + 1),
                            max(0, nearestIntKeypoint[2] - int(sizeOfCircularWindow/2)):min(diff_of_gauss.shape[2], nearestIntKeypoint[2] + int(sizeOfCircularWindow/2) + 1)]
        magnitudeArea = magnitude[max(0, nearestIntKeypoint[1] - int(sizeOfCircularWindow/2)):min(diff_of_gauss.shape[1], nearestIntKeypoint[1] + int(sizeOfCircularWindow/2) + 1),
                                max(0, nearestIntKeypoint[2] - int(sizeOfCircularWindow/2)):min(diff_of_gauss.shape[2], nearestIntKeypoint[2] + int(sizeOfCircularWindow/2) + 1)]

        magnitudeArea = np.multiply(magnitudeArea, cv2.GaussianBlur(magnitudeArea, (0, 0),
                                        sigmaX=1.5 * sigma*(k**nearestIntKeypoint[0]),
                                        sigmaY=1.5 * sigma*(k**nearestIntKeypoint[0])))
        vec = np.zeros(36)
        
        # For each cell, divide angle by 10 to find out cell it's in
        bins = anglesArea/10

        # Create the amount they weigh towards the lower bin and the upper bin.
        # If a value is an integer, its fully under weightsTowardsLower. So if a
        # value of an element is 2, then the corresponding values are 1 and 0.

        weightsTowardsLower = (10*(np.floor(bins)+1)-anglesArea)/10

        addToLowerBins = magnitudeArea*weightsTowardsLower
        addToUpperBins = magnitudeArea - addToLowerBins

        lowerBins = np.uint8(np.floor(bins))
        upperBins = np.uint8(np.ceil(bins) % 36)

        np.add.at(vec, lowerBins, addToLowerBins)
        np.add.at(vec, upperBins, addToUpperBins)

        indexWithMost = np.argmax(vec)
        moreThan80 = vec >= 0.8*vec[indexWithMost]
        arrOfOrientations = []
        for j in range(0, len(vec)):
            if moreThan80[j]:
                arrOfOrientations.append(j*10)
        mapOfKeypointsToOrientation[i] = np.array(arrOfOrientations)
    return mapOfKeypointsToOrientation


##################
# Main Functions #
##################


def paper_main_function_SIFT(img):
    """
    Creates a SIFT face descriptor without finding the keypoints and just
    getting the descriptors for each patch in the image.

    Keyword Arguments:
    - img: The face image that you want the SIFT vector for

    Returns:
    - A 6272 long vector for the face image if it is 64x64. 
    """
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grid_square_size = (int(gray_image.shape[0]/16), int(gray_image.shape[1]/16))
    number_of_squares_per_patch = (int((gray_image.shape[0]/8)/grid_square_size[0]),
                                   int((gray_image.shape[1]/8)/grid_square_size[1]))
    size_of_patch = (number_of_squares_per_patch[0]*grid_square_size[0],
                     number_of_squares_per_patch[1]*grid_square_size[1])
    vec = np.array([])

    # Gaussian weighing "function" based on how far the center of a square is
    # from the center of the patch and with sigma = 1
    # The constant in front is not here since the vector will be normalized so
    # that provides nothing.
    gaussian_weighing = np.exp([-1, -1.5, -4])

    # For a 64 x 64, want to examine a 8x8 square and have the squares 4 apart.
    for i in range(0, 7):
        for j in range(0, 7):
            vecForPatch = np.array([])
            patch = gray_image[i*size_of_patch[0]:(i + 1)*size_of_patch[0],
                               j*size_of_patch[1]:(j + 1)*size_of_patch[1]]
            gx = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=1)
            gy = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=1)
            magnitude, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
            size_of_small_square = (int(size_of_patch[0]/4), int(size_of_patch[1]/4))
            # Break up patch into 4x4
            for row in range(0, 4):
                for col in range(0, 4):
                    gaussian_weight = gaussian_weighing[int(np.abs(np.array([row, col]) - 1.5).sum()) - 1]
                    vecSquare = np.zeros(8)
                    magSquare = magnitude[row*size_of_small_square[0]:(row+1)*size_of_small_square[0],
                                          col*size_of_small_square[1]:(col+1)*size_of_small_square[1]]
                    angleSquare = angle[row*size_of_small_square[0]:(row+1)*size_of_small_square[0],
                                        col*size_of_small_square[1]:(col+1)*size_of_small_square[1]]
                    bins = angleSquare/45
                    weightsTowardsLower = (45*(np.floor(bins)+1)-angleSquare)/45
                    magSquare = gaussian_weight * magSquare

                    addToLowerBins = magSquare*weightsTowardsLower
                    addToUpperBins = magSquare - addToLowerBins

                    lowerBins = np.uint8(np.floor(bins))
                    upperBins = np.uint8(np.ceil(bins) % 8)

                    np.add.at(vecSquare, lowerBins, addToLowerBins)
                    np.add.at(vecSquare, upperBins, addToUpperBins)
                    vecForPatch = np.append(vecForPatch, vecSquare)
            norm = np.linalg.norm(vecForPatch)
            if norm != 0: 
                vecForPatch = vecForPatch / norm
            vec = np.append(vec, vecForPatch)
    return vec
