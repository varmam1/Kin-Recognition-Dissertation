SHELL := /bin/bash

init:
	pip3 install -r requirements.txt

runTests:
	py.test src/test

coverage:
	coverage run -m py.test src/test && coverage report -m

faceDescriptorsKFW1:
	python3 -m src.scripts.get_fds_and_save "KinFaceW-I"

faceDescriptorsKFW2:
	python3 -m src.scripts.get_fds_and_save "KinFaceW-II"

runWGEMLKFW1Unrestricted:
	python3 -m src.scripts.run_WGEML "KinFaceW-I" "fd" "unrestricted"
	python3 -m src.scripts.run_WGEML "KinFaceW-I" "fs" "unrestricted"
	python3 -m src.scripts.run_WGEML "KinFaceW-I" "md" "unrestricted"
	python3 -m src.scripts.run_WGEML "KinFaceW-I" "ms" "unrestricted"

runWGEMLKFW1Restricted:
	python3 -m src.scripts.run_WGEML "KinFaceW-I" "fd" "restricted"
	python3 -m src.scripts.run_WGEML "KinFaceW-I" "fs" "restricted"
	python3 -m src.scripts.run_WGEML "KinFaceW-I" "md" "restricted"
	python3 -m src.scripts.run_WGEML "KinFaceW-I" "ms" "restricted"

runWGEMLKFW2Unrestricted:
	python3 -m src.scripts.run_WGEML "KinFaceW-II" "fd" "unrestricted"
	python3 -m src.scripts.run_WGEML "KinFaceW-II" "fs" "unrestricted"
	python3 -m src.scripts.run_WGEML "KinFaceW-II" "md" "unrestricted"
	python3 -m src.scripts.run_WGEML "KinFaceW-II" "ms" "unrestricted"

runWGEMLKFW2Restricted:
	python3 -m src.scripts.run_WGEML "KinFaceW-II" "fd" "restricted"
	python3 -m src.scripts.run_WGEML "KinFaceW-II" "fs" "restricted"
	python3 -m src.scripts.run_WGEML "KinFaceW-II" "md" "restricted"
	python3 -m src.scripts.run_WGEML "KinFaceW-II" "ms" "restricted"

clean:
	find -iname "*.pyc" -delete
	find -iname "__pycache__" -delete
