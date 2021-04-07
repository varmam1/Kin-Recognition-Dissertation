import pytest
import numpy as np
import os
from ...face_descriptors import CifarNet

def test_create_CFN_model():
    model = CifarNet.define_CifarNet_model()
    inputShapeCorrect = model.layers[0].get_input_shape_at(0) == (None, 32, 32, 3)
    outputShapeCorrect = model.layers[-1].get_output_shape_at(0) == (None, 10)
    assert len(model.layers) == 21 and outputShapeCorrect and inputShapeCorrect


def test_load_CIFAR_10():
    (trainX, trainY), (testX, testY) = CifarNet.load_CIFAR_10_data()
    assert trainX.shape == (50000, 32, 32, 3) and trainY.shape == (50000, 10) and testX.shape == (10000, 32, 32, 3) and testY.shape == (10000, 10)


def test_create_CFN_model_for_face_description():
    model = CifarNet.get_CFN_face_descriptor_model()
    inputShapeCorrect = model.layers[0].get_input_shape_at(0) == (None, 32, 32, 3)
    outputShapeCorrect = model.layers[-1].get_output_shape_at(0) == (None, 192)
    assert len(model.layers) == 21 and outputShapeCorrect and inputShapeCorrect
