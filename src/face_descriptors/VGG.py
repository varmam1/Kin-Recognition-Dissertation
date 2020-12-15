import numpy as np
import cv2
import tensorflow as tf
from keras.layers import Flatten, Dense, Input, Activation, Conv2D, MaxPooling2D, Dropout
from keras.models import Sequential, Model, load_model
from scipy.io import loadmat
import urllib.request
import tarfile
import os
import shutil


# The file which will take a face and return the 4096-dimensional VGG vector
# Using the VGG-Very-Deep-16 CNN architecture

MATCONVNET_WEIGHTS_PATH = "https://www.robots.ox.ac.uk/~vgg/software/vgg_face/src/vgg_face_matconvnet.tar.gz"
VGG_WEIGHTS_PATH = 'src/face_descriptors/vgg-face-my-model.h5'


def create_model(weights_path=None):
    """
    Returns the architecture of the VGG16 model without the softmax layer.
    """
    model = Sequential()
    model.add(Input(shape=(224, 224, 3)))

    # Block 1
    model.add(Conv2D(64, (3, 3), activation='relu',
                     padding='same', name='conv1_1'))
    model.add(Conv2D(64, (3, 3), activation='relu',
                     padding='same', name='conv1_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='pool1'))

    # Block 2
    model.add(Conv2D(128, (3, 3), activation='relu',
                     padding='same', name='conv2_1'))
    model.add(Conv2D(128, (3, 3), activation='relu',
                     padding='same', name='conv2_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='pool2'))

    # Block 3
    model.add(Conv2D(256, (3, 3), activation='relu',
                     padding='same', name='conv3_1'))
    model.add(Conv2D(256, (3, 3), activation='relu',
                     padding='same', name='conv3_2'))
    model.add(Conv2D(256, (3, 3), activation='relu',
                     padding='same', name='conv3_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='pool3'))

    # Block 4
    model.add(Conv2D(512, (3, 3), activation='relu',
                     padding='same', name='conv4_1'))
    model.add(Conv2D(512, (3, 3), activation='relu',
                     padding='same', name='conv4_2'))
    model.add(Conv2D(512, (3, 3), activation='relu',
                     padding='same', name='conv4_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='pool4'))

    # Block 5
    model.add(Conv2D(512, (3, 3), activation='relu',
                     padding='same', name='conv5_1'))
    model.add(Conv2D(512, (3, 3), activation='relu',
                     padding='same', name='conv5_2'))
    model.add(Conv2D(512, (3, 3), activation='relu',
                     padding='same', name='conv5_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='pool5'))

    # Face Descriptor Block
    model.add(Flatten(name='flatten'))
    model.add(Dense(4096, activation='relu', name='fc6'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu', name='fc7'))
    model.add(Dropout(0.5))
    model.add(Dense(2622, activation='softmax', name='fc8'))

    if weights_path != None:
        model.load_weights(weights_path)

    return model


def save_weights_from_vgg_face_online(path_for_vgg_weights):
    """
    Downloads the matconvnet weights from
    https://www.robots.ox.ac.uk/~vgg/software/vgg_face/ and modifies it
    slightly to match our model, saves the modified weights and deletes the
    downloaded files.
    """
    # Downloads and loads the weights from the site
    urllib.request.urlretrieve(
        MATCONVNET_WEIGHTS_PATH, 'vgg_face_matconvnet.tar.gz')
    tar = tarfile.open("vgg_face_matconvnet.tar.gz", "r:gz")
    tar.extractall("weights")
    tar.close()
    data = loadmat('weights/vgg_face_matconvnet/data/vgg_face.mat',
                   matlab_compatible=False, struct_as_record=False)
    net = data["net"][0][0]
    ref_model_layers = net.layers[0]

    # Creates the architecture that will be used
    model = create_model()
    layer_names = [layer.name for layer in model.layers]
    num_of_ref_model_layers = ref_model_layers.shape[0]

    # For each layer, if it is a conv2D or fully-connected layer, applies the
    # weights as required
    for i in range(num_of_ref_model_layers):
        ref_model_layer = ref_model_layers[i][0][0].name[0]
        if ref_model_layer in layer_names:
            if ref_model_layer.find("conv") == 0 or ref_model_layer.find("fc") == 0:
                base_model_index = layer_names.index(ref_model_layer)
                weights = ref_model_layers[i][0][0].weights[0, 0]
                bias = ref_model_layers[i][0][0].weights[0, 1]

                # fc6 is of the form (7, 7, 512, 4096) originally but is
                # required to be of the form (7*7*512, 4096) for this
                if (ref_model_layer == "fc6"):
                    weights = np.reshape(weights, (7*7*512, 4096))

                # The other fc layers are of the form (1, 1, 4096, x)
                # and we need (4096, x)
                elif (ref_model_layer.find("fc") == 0):
                    weights = weights[0][0]

                model.layers[base_model_index].set_weights(
                    [weights, bias[:, 0]])

    # Saves the weights to disk
    model.save_weights(path_for_vgg_weights)

    # Removes the downloaded files
    os.remove("vgg_face_matconvnet.tar.gz")
    shutil.rmtree("weights")


def create_face_descriptor_model_with_weights():
    """
    A wrapper function to create the necessary model with all of the necessary
    weights. It downloads the weights from online and saves it in the proper
    format if the weights aren't in the proper place and, otherwise, creates
    the model loaded with the necessary weights.

    Returns:
    - A keras Sequential model which is the VGG network loaded with the
    weights to recognize faces.
    """
    if not os.path.exists(VGG_WEIGHTS_PATH):
        save_weights_from_vgg_face_online(VGG_WEIGHTS_PATH)
    model = create_model(VGG_WEIGHTS_PATH)
    return Model(inputs=model.input, outputs=model.layers[-2].output)


def get_VGG_face_descriptor(imgs):
    """
    Given the image and the VGG model, returns the 4096-dimensional vector
    that corresponds to the face.

    Keyword Arguments:
    - imgs (np.array) : An np array that represents a list of face images.
    Shape of (n, x, x, 3)

    Returns:
    - An np array of shape (n, 4096) representing the VGG face descriptor for
    each corresponding image.
    """
    model = create_face_descriptor_model_with_weights()
    resized_imgs = np.zeros((imgs.shape[0], 224, 224, 3))
    for i, img in enumerate(imgs):
        resized_imgs[i, :, :, :] = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
    return model.predict(resized_imgs)
