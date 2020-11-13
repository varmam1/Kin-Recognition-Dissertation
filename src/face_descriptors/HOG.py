import numpy as np
import cv2

# Create a 2880-dim vector by having 9 dim vector for 16x16 blocks then for 8x8 blocks

def compute_gradients(img):
    """
    Given a colored image, computes the gradients for each point and returns the magnitude
    and angle for each point. 

    Keyword Arguments:
    - img (np.array): An np array that represents the colored image 

    Returns:
    - A tuple of the magnitude and angle np arrays. 
        - magnitude (np.array): An array representing the magnitude of the gradient at each pixel
        - angle (np.array): An array representing the angle of the gradient with respect to the x axis
                            at each pixel.
    """
    img = np.float32(img)/ 255.0
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)
    magnitude, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    return (magnitude, angle)