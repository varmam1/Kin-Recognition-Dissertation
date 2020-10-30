import numpy as np
import cv2

image = cv2.imread("test.jpg")

# TODO: Create function that saves each face as it's own image rather than just adds the rectangle to the image. 

def draw_rectangle_on_faces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # faceCascade imports in the previously made classifier
    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = faceCascade.detectMultiScale(
        gray,     
        scaleFactor=1.2,
        minNeighbors=1,     
        minSize=(100, 100)
    )

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

draw_rectangle_on_faces(image)

cv2.imwrite('out.jpg', image)
