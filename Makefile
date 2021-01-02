SHELL := /bin/bash

init:
	pip3 install -r requirements.txt

runTests:
	py.test src/test

coverage:
	coverage run -m py.test src/test && coverage report -m

faceDescriptorsKFW1:
	python3 -m src.create_face_descriptors.get_fds_and_save "KinFaceW-I"

faceDescriptorsKFW2:
	python3 -m src.create_face_descriptors.get_fds_and_save "KinFaceW-II"

clean:
	find -iname "*.pyc" -delete
	find -iname "__pycache__" -delete
