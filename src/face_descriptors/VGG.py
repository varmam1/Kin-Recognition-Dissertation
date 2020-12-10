import numpy as np
import cv2
from keras.layers import Flatten, Dense, Input, Activation, Conv2D, MaxPooling2D
from keras.models import Sequential

# The file which will take a face and return the 4096-dimensional VGG vector
# Using the VGG-Very-Deep-16 CNN architecture

def create_model(weights_path=None):
    """
    Returns the architecture of the VGG16 model without the softmax layer.
    """
    model = Sequential()
    model.add(Input(shape=(224, 224, 3)))

    # Block 1
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_1'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='pool1'))

    # Block 2
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_1'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='pool2'))

    # Block 3
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_1'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_2'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='pool3'))

    # Block 4
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_1'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_2'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='pool4'))

    # Block 5
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_1'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_2'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='pool5'))

    # Face Descriptor Block
    model.add(Flatten(name='flatten'))
    model.add(Dense(4096, name='fc6'))
    model.add(Dense(4096, name='fc7'))

    if weights_path != None:
        model.load_weights(weights_path)

    return model

def get_VGG_face_descriptor(img, model):
    """
    Given the image and the VGG model, returns the 4096-dimensional vector
    that corresponds to the face.

    Keyword Arguments:
    - img (np.array) : An np array that represents the face image. Shape of
      (x, x, 64)
    - model: 

    Returns:
    - A length 4096 np array representing the VGG face descriptor.
    """
