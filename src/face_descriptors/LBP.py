
import numpy as np
import cv2

# The file which will take a face and return the 3776-dimensional vector of the face that is 64 x 64 pixels. 

def get_number_from_neighborhood(neighborhood):
    """
    Given a 3x3 neighborhood as a numpy array, returns the corresponding LBP value for the neighborhood. 
    """
    greaterOrLess = (neighborhood > neighborhood[1, 1]).astype(int)
    return (greaterOrLess[1, 0]*(2**7) + greaterOrLess[2, 0]*(2**6)
        + greaterOrLess[2, 1]*(2**5) + greaterOrLess[2, 2]*(2**4) 
        + greaterOrLess[1, 2]*(2**3) + greaterOrLess[0, 2]*(2**2) 
        + greaterOrLess[0, 1]*(2) + + greaterOrLess[0, 0])
