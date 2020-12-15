import pytest
import numpy as np
import os
from ...face_descriptors import VGG


def test_create_model_with_no_weights():
    model = VGG.create_model()
    assert len(model.layers) == 24


# The following test takes a long time so only uncomment if you want to run it
def test_get_weights_from_online():
    VGG.save_weights_from_vgg_face_online("vgg-face-test.h5")
    exists = os.path.exists("vgg-face-test.h5")
    if exists:
        os.remove("vgg-face-test.h5")
    assert exists


def test_create_face_descriptor_model():
    model = VGG.create_model("src/face_descriptors/vgg-face-my-model.h5")
    face_model = VGG.create_face_descriptor_model_with_weights()
    important_layers = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2',
                        'conv3_1', 'conv3_2', 'conv3_3', 'conv4_1',
                        'conv4_2', 'conv4_3', 'conv5_1', 'conv5_2',
                        'conv5_3', 'fc6', 'fc7']
    same_weights = True
    for layer in important_layers:
        same_weights = same_weights and (model.get_layer(layer).get_weights()[0] == face_model.get_layer(layer).get_weights()[0]).all()
        same_weights = same_weights and (model.get_layer(layer).get_weights()[1] == face_model.get_layer(layer).get_weights()[1]).all()
    assert same_weights
