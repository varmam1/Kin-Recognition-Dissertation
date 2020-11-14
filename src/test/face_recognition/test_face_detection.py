import pytest
import cv2
import numpy as np
from ...face_recognition import face_detection as fd

# =================== Testing Uniform Vals Function ===================
def test_face_detector():
    img = cv2.imread("src/test/face_recognition/test.jpg")
    faces = fd.face_detector(img)
    expectedFace = np.array([(529, 272, 247, 247)])
    # Add some leeway
    assert np.array_equal(expectedFace, faces)

# ===================== Testing Resize Function =====================

def test_resize_to_64_by_64_pixels():
    img = cv2.imread("src/test/face_recognition/test.jpg")
    assert fd.resize_img(img, 64, 64).shape == (64, 64, 3)

