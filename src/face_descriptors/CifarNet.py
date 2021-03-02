import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from keras.layers import Flatten, Dense, Input, Activation, Conv2D, MaxPooling2D, Dropout, BatchNormalization, AveragePooling2D, Reshape
from keras.models import Sequential, Model, load_model
from keras.initializers import *
from keras.regularizers import *
from keras.optimizers import SGD


def define_CifarNet_model(num_classes=10, lr=0.001):
    """
    Creates a CNN which takes in a 32 x 32 x 3 input and has output shape of
    num_classes.

    Optional Arguments:
    - num_classes: The number of classes there are in the dataset if not using CIFAR-10
    - lr: The learning rate for stochastic gradient descent

    Returns:
    - The corresponding keras model compiled with the optimizer and loss function
    """
    model = Sequential()
    model.add(Input(shape=(32, 32, 3)))
    
    model.add(Conv2D(64, (3, 3), kernel_initializer=he_uniform(), kernel_regularizer=l2(l=0.0001)))
    model.add(BatchNormalization(momentum=0.9997))

    model.add(Conv2D(64, (3, 3), padding="same", kernel_initializer=he_uniform(), kernel_regularizer=l2(l=0.0001)))
    model.add(BatchNormalization(momentum=0.9997))
    
    model.add(Conv2D(128, (3, 3), strides=(2, 2), padding="same", kernel_initializer=he_uniform(), kernel_regularizer=l2(l=0.0001)))
    model.add(BatchNormalization(momentum=0.9997))
    
    model.add(Conv2D(128, (3, 3), padding="same", kernel_initializer=he_uniform(), kernel_regularizer=l2(l=0.0001)))
    model.add(BatchNormalization(momentum=0.9997))
    model.add(Dropout(0.5))
    
    model.add(Conv2D(128, (3, 3), padding="same", kernel_initializer=he_uniform(), kernel_regularizer=l2(l=0.0001)))
    model.add(BatchNormalization(momentum=0.9997))
    
    model.add(Conv2D(192, (3, 3), strides=(2, 2), padding="same", kernel_initializer=he_uniform(), kernel_regularizer=l2(l=0.0001)))
    model.add(BatchNormalization(momentum=0.9997))
    
    model.add(Conv2D(192, (3, 3), padding="same", kernel_initializer=he_uniform(), kernel_regularizer=l2(l=0.0001)))
    model.add(BatchNormalization(momentum=0.9997))
    model.add(Dropout(0.5))
    
    model.add(Conv2D(192, (3, 3), padding="same", kernel_initializer=he_uniform(), kernel_regularizer=l2(l=0.0001)))
    model.add(BatchNormalization(momentum=0.9997))
    
    model.add(AveragePooling2D(pool_size=(8, 8)))
    model.add(Dense(num_classes))
    model.add(Reshape([num_classes]))
    model.compile(optimizer=SGD(lr=lr, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def preprocess_data(x, y):
    """
    Given the X and Y values, pre-processes the CIFAR-10 dataset using the
    densenet model requirements.

    Keyword Arguments:
    - X: A numpy array of shape (n, 32, 32, 3) which has n images
    - Y: A numpy array of shape (n, 1) which has n labels

    Returns:
    - The X values properly preprocessed and the y values as one-hot encoding
    """
    x = keras.applications.densenet.preprocess_input(x)
    y = keras.utils.to_categorical(y)
    return x, y


def load_CIFAR_10_data():
    """
    Returns the training/testing split for the CIFAR-10 dataset. 
    """
    (trainX, trainY), (testX, testY) = cifar10.load_data()
    trainX, trainY = preprocess_data(trainX, trainY)
    testX, testY = preprocess_data(testX, testY)
    return (trainX, trainY), (testX, testY)


def train_model(trainX, trainY, testX, testY, model, numEpochs=100, batch_size=64, save_location="CifarNet.h5"):
    """
    Given the dataset and the model, trains the model and saves the weights on disk. 

    Keyword Arguments:
    - trainX: The training set of 32 x 32 pixel images
    - trainY: The training set of classifications of the images
    - testX: The test set of 32 x 32 pixel images
    - testY: The test set of classifications of the images
    - model: The Keras model to train

    Optional Arguments:
    - numEpochs: The amount of times SGD should be run for
    - batch_size: The batch size for SGD
    - save_location: Where the weights should be saved on disk
    """

    model.fit(trainX, trainY, epochs=numEpochs, batch_size=batch_size, validation_data=(testX, testY))
    model.save(save_location)


def create_model_and_train_and_test():
    model = define_CifarNet_model()
    (trainX, trainY), (testX, testY) = load_CIFAR_10_data()
    train_model(trainX, trainY, testX, testY, model)

    results = model.evaluate(testX, testY)
    print("Test Loss, Test Accuracy:", results)

