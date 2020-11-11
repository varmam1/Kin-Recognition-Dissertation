import numpy as np
import cv2

# The file which will take a face and return the 3776-dimensional vector of the face that is 64 x 64 pixels. 

def generate_uniform_values():
    """
    Creates an array of the uniform values from 0 to 255. 

    Returns:
    - A numpy array that contains only the uniform values from 0 to 255. Size of 58. 
    """
    uniform = np.array([])
    for i in range(0, 256):
        numChanges = 0
        binaryString = "{0:b}".format(i).zfill(8)
        lastVal = binaryString[0]
        for j in range(1, 7):
            if (binaryString[j] != lastVal):
                numChanges += 1
                lastVal = binaryString[j]
        if binaryString[7] != binaryString[0]:
            numChanges += 1
        if numChanges < 3:
            uniform = np.append(uniform, [i])
    return uniform

def get_number_from_neighborhood(neighborhood, uniform_vals):
    """
    Given a 3x3 neighborhood as a numpy array, returns the corresponding LBP value for the neighborhood. 
    This only returns the value if the it is uniform (if it contains at most two bitwise transitions from
    0 to 1 or vice versa when the bit pattern is traversed circularly) and if it isn't, it returns -1. 

    Keyword Arguments:
    - neighborhood: A 3x3 numpy array representing the neighborhood of a pixel
    - uniform_vals: A numpy array of the uniform values from 0 to 256. 

    Returns:
    - The corresponding LBP value for the pixel if it is uniform and -1 if it isn't. 
    """
    greaterOrLess = (neighborhood > neighborhood[1, 1]).astype(int)
    val = (greaterOrLess[1, 0]*(2**7) + greaterOrLess[2, 0]*(2**6)
        + greaterOrLess[2, 1]*(2**5) + greaterOrLess[2, 2]*(2**4) 
        + greaterOrLess[1, 2]*(2**3) + greaterOrLess[0, 2]*(2**2) 
        + greaterOrLess[0, 1]*(2) + + greaterOrLess[0, 0])
    
    if val in uniform_vals:
        return val
    return -1

def LBP_image(gray_image, uniform_vals):
    """
    Given a grayscaled image, returns the LBP image. 

    Keyword Arguments:
    - gray_image: A numpy array of the image that the LBPs needs to be calculated for. 
    - uniform_vals: A numpy array of the uniform values from 0 to 256. 

    Returns:
    - A numpy array of the same shape with each value in the corresponding pixel. 
    """
    padded_gray_image = np.pad(gray_image, (1, 1), 'constant')
    newImg = np.zeros(gray_image.shape)
    for i in range(1, gray_image.shape[0] + 1):
        for j in range(1, gray_image.shape[1] + 1):
            val = get_number_from_neighborhood(padded_gray_image[i - 1:i + 2, j - 1:j + 2], uniform_vals)
            newImg[i - 1, j - 1] = val
    return newImg

def create_LBP_feature_vector(image):
    """
    Given an image, this method will create the LBP descriptor for the image as a numpy array. 

    For a 64 x 64 pixel image, this will create 8 x 8 blocks and go through each of the blocks 
    on each row. Each block creates a 59 dimensional vector which is the histogram for the 
    values of each of the pixels. The counts correspond to the values -1 :: uniform_values
    so the first count is how many non-uniform values there are, then how many 0s there are, etc.
    
    Each individual vector is then appended together into one 3776 dimensional vector. 

    Keyword Arguments: 
    - image: The image that you want the LBP descriptor of. 

    Returns:
    - A numpy array which is the LBP descriptor. 
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    uniform_values = generate_uniform_values()
    LBPs = LBP_image(gray_image, uniform_values)

    # Prepend the non-uniform case to the beginning for all values
    all_values = np.append([-1], uniform_values)
    feature_vector = np.array([])

    for row in range(0, int(gray_image.shape[0]/8)):
        for col in range(0, int(gray_image.shape[1]/8)):        
            block = LBPs[row * 8: (row + 1) * 8, col * 8: (col + 1) * 8]
            histogram = np.zeros(59)

            for (i, val) in enumerate(all_values):
                histogram[i] = (block == val).astype(int).sum()
            
            feature_vector = np.append(feature_vector, histogram)
    
    return feature_vector
