import numpy as np
import cv2

# In our experiments, if the number of frames of a video is more than 100, we just randomly detected 100 frames of this video


def get_random_frames(face_video, amnt=100):
    """
    Given a video feed of a face, returns 100 random frames to get the face
    descriptors from. 

    Keyword Arguemnts:
    - face_video: A (n, 64, 64, 3) shape numpy array which represents a face
        over n frames. 
    - amnt: The amount of frames that should be returned

    Returns:
    - A (amnt, 64, 64, 3) numpy array which is the $amnt frames. 
    """
    return face_video[np.random.choice(len(face_video), size=amnt, replace=False)]


def get_specified_face_descriptor(frames, face_descriptor):
    """
    Given the frames of the face and a function which takes an image and
    returns the face descriptor of it, returns the average face descriptor
    for the face.

    Keyword Arguments:
    - frames: A (m, 64, 64, 3) array which represents the m frames of a face
    - face_descriptor: A function that takes (64, 64, 3) numpy array -> 
        (n, ) numpy array describing the image.
    
    Returns:
    - An (n, ) numpy array which is the average of the face descriptor over
        all frames.
    """
    fds = []
    for img in frames:
        fds.append(face_descriptor(img))
    fds = np.array(fds)
    return np.mean(fds, axis=0)
