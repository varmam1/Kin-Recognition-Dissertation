import numpy as np
import cv2

# Create a 2880-dim vector by having 9 dim vector for 16x16 blocks then for 8x8 blocks

def compute_gradients(img):
    """
    Given a colored image, computes the unsigned gradients for each point and returns the magnitude
    and angle for each point. 

    Keyword Arguments:
    - img (np.array): An np array that represents the colored image 

    Returns:
    - A tuple of the magnitude and angle np arrays. 
        - magnitude (np.array): An array representing the magnitude of the gradient at each pixel
        - angle (np.array): An array representing the angle of the gradient with respect to the x axis
                            at each pixel. Ranges from 0 to 180.
    """
    img = np.float32(img)/ 255.0
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)
    magnitude, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)

    # If the image is colored, take the max for each pixel along the 3 channels
    if len(img.shape) == 3:
        magnitude = np.amax(magnitude, axis=(2))
        angle = np.amax(angle, axis=(2))
    
    angle = angle % 180
    return (magnitude, angle)

def create_block_HOG_vector(blockMagnitudes, blockAngles):
    """
    Given the magnitudes and angles of the gradients of a block of an image, returns the 
    9-dimensional histogram for the block. The counts for each angle are weighted by the magnitude
    and by how far the angle is from the necessary angles. For example, if a pixel had a gradient with 
    angle 30 and magnitude 2, it would contribute 1 to the 20 bin and 1 to the 40 bin. 

    Keyword Arguments:
    - blockMagnitudes (np.array) : A numpy array of shape (w, h) which has the magnitudes of the gradient for each pixel in the block.
    - blockAngles (np.array) : A numpy array of shape (w, h) which has the unsigned angles of the gradient for each pixel in the block.

    Returns:
    - A numpy array of shape (9, ) which is the histogram of the given block as defined above. 
    """
    vec = np.zeros(9)
    # For each cell, divide angle by 20 to find out what cell it's supposed to be in
    bins = blockAngles/20
    # Create the amount they weigh towards the lower bin and the upper bin. If a value is an integer, its fully unde
    # weightsTowardsLower. So if a value of an element is 2, then the corresponding values are 1 and 0 for each.
    weightsTowardsLower = (20*(np.floor(bins)+1)-blockAngles)/20
    addToLowerBins = blockMagnitudes*weightsTowardsLower
    addToUpperBins = blockMagnitudes - addToLowerBins
    lowerBins = np.uint8(np.floor(bins))
    upperBins = np.uint8(np.ceil(bins) % 9)
    np.add.at(vec, lowerBins, addToLowerBins)
    np.add.at(vec, upperBins, addToUpperBins)
    return vec