import numpy as np
import cv2

def face_detector(img):
    """
    Given an image, returns a list of tuples (x, y, w, h) where (x, y) is the coordinate of the top left vertex and
    (w, h) represents the width and height of the face. 

    Keyword Arguments:
    - img: The image that the faces are going to obtained from.

    Returns:
    - A list of tuples describing the faces in the form described above. 
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # faceCascade imports in the previously made classifier
    faceCascade = cv2.CascadeClassifier('src/face_detection/haarcascade_frontalface_default.xml')
    faces = faceCascade.detectMultiScale(
        gray,     
        scaleFactor=1.2,
        minNeighbors=1,     
        minSize=(100, 100)
    )

    return faces

def resize_img(img, width, height):
    """
    Given the image and the target width and height, returns the image resized to the specifications.

    Keyword Arguments:
    - img: The image that is going to be resized.
    - width (int): Target width in pixels
    - height (int): Target height in pixels

    Returns:
    - The image resized. 
    """
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

def save_image(img, path):
    """
    Given the image and an output path, saves the image to the path.

    Keyword Arguments:
    - img: The image that is going to be saved.
    - path (String): The path to the place to save the image
    """
    cv2.imwrite(path, img)

def save_faces(img, faces, width, height):
    """
    Saves the faces in an image to an external image with the size of the image specified.

    Keyword Arguments:
    - img: The image that has the faces that are going to be saved.
    - faces (List[(Int, Int, Int, Int)]): A list of faces specified by the top left vertex and the width and height of the face
    - width (int): Target width in pixels for the size of the image
    - height (int): Target height in pixels for the size of the image
    """

    for i, (x, y, w, h) in enumerate(faces):
        # TODO: Change output string
        save_image(resize_img(img[y:y+h, x:x+w], width, height), 'out' + str(i) + '.jpg')


def save_faces_in_64_by_64(image):
    """
    Given an image which is an np.array, saves the faces in the image in a file with dimenesions of 64x64. 

    Keyword Arguments:
    - image: An np.array which is of shape (w, h, 3) which is a representation of the image. 
    """
    save_faces(image, face_detector(image), 64, 64)
