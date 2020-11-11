SHELL := /bin/bash

init:
	pip3 install -r requirements.txt

runTests:
	py.test src/test

coverage:
	coverage run -m py.test src/test && coverage report -m

testLBP:
	python3 -m src.face_descriptors.LBP

clean:
	find -iname "*.pyc" -delete
	find -iname "__pycache__" -delete
