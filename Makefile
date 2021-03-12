SHELL := /bin/bash

init:
	pip3 install -r requirements.txt

###############################
######## Test Targets #########
###############################
runTests:
	py.test src/test

coverage:
	coverage run -m py.test src/test && coverage report -m


#################################
# Face Descriptor Maker Targets #
#################################

faceDescriptorsKFW1:
	python3 -m src.scripts.get_fds_and_save "KinFaceW-I"

faceDescriptorsKFW2:
	python3 -m src.scripts.get_fds_and_save "KinFaceW-II"

faceDescriptorsTSK:
	python3 -m src.scripts.get_fds_and_save "TSKinFace"

getAllFaceDescriptors:
	make faceDescriptorsKFW1
	make faceDescriptorsKFW2
	make faceDescriptorsTSK


#################################
###### WGEML Runner Targets #####
#################################


runWGEMLKFW1Unrestricted:
	python3 -m src.scripts.run_WGEML "KinFaceW-I" "fs" "unrestricted"
	python3 -m src.scripts.run_WGEML "KinFaceW-I" "fd" "unrestricted"
	python3 -m src.scripts.run_WGEML "KinFaceW-I" "ms" "unrestricted"
	python3 -m src.scripts.run_WGEML "KinFaceW-I" "md" "unrestricted"

runWGEMLKFW1Restricted:
	python3 -m src.scripts.run_WGEML "KinFaceW-I" "fs" "restricted"
	python3 -m src.scripts.run_WGEML "KinFaceW-I" "fd" "restricted"
	python3 -m src.scripts.run_WGEML "KinFaceW-I" "ms" "restricted"
	python3 -m src.scripts.run_WGEML "KinFaceW-I" "md" "restricted"

runWGEMLKFW2Unrestricted:
	python3 -m src.scripts.run_WGEML "KinFaceW-II" "fs" "unrestricted"
	python3 -m src.scripts.run_WGEML "KinFaceW-II" "fd" "unrestricted"
	python3 -m src.scripts.run_WGEML "KinFaceW-II" "ms" "unrestricted"
	python3 -m src.scripts.run_WGEML "KinFaceW-II" "md" "unrestricted"

runWGEMLKFW2Restricted:
	python3 -m src.scripts.run_WGEML "KinFaceW-II" "fs" "restricted"
	python3 -m src.scripts.run_WGEML "KinFaceW-II" "fd" "restricted"
	python3 -m src.scripts.run_WGEML "KinFaceW-II" "ms" "restricted"
	python3 -m src.scripts.run_WGEML "KinFaceW-II" "md" "restricted"

runWGEMLTSK:
	python3 -m src.scripts.run_WGEML "TSKinFace" "fs" "null"
	python3 -m src.scripts.run_WGEML "TSKinFace" "fd" "null"
	python3 -m src.scripts.run_WGEML "TSKinFace" "ms" "null"
	python3 -m src.scripts.run_WGEML "TSKinFace" "md" "null"
	python3 -m src.scripts.run_WGEML "TSKinFace" "fms" "null"
	python3 -m src.scripts.run_WGEML "TSKinFace" "fmd" "null"

runWGEML:
	make runWGEMLKFW1Unrestricted
	make runWGEMLKFW1Restricted
	make runWGEMLKFW2Unrestricted
	make runWGEMLKFW2Restricted
	make runWGEMLTSK

#################################
####### Predictor Targets #######
#################################

runPredictionKFW1Unrestricted:
	python3 -m src.scripts.get_accuracies "KinFaceW-I" "fs" "unrestricted"
	python3 -m src.scripts.get_accuracies "KinFaceW-I" "fd" "unrestricted"
	python3 -m src.scripts.get_accuracies "KinFaceW-I" "ms" "unrestricted"
	python3 -m src.scripts.get_accuracies "KinFaceW-I" "md" "unrestricted"

runPredictionKFW1Restricted:
	python3 -m src.scripts.get_accuracies "KinFaceW-I" "fs" "restricted"
	python3 -m src.scripts.get_accuracies "KinFaceW-I" "fd" "restricted"
	python3 -m src.scripts.get_accuracies "KinFaceW-I" "ms" "restricted"
	python3 -m src.scripts.get_accuracies "KinFaceW-I" "md" "restricted"

runPredictionKFW2Unrestricted:
	python3 -m src.scripts.get_accuracies "KinFaceW-II" "fs" "unrestricted"
	python3 -m src.scripts.get_accuracies "KinFaceW-II" "fd" "unrestricted"
	python3 -m src.scripts.get_accuracies "KinFaceW-II" "ms" "unrestricted"
	python3 -m src.scripts.get_accuracies "KinFaceW-II" "md" "unrestricted"

runPredictionKFW2Restricted:
	python3 -m src.scripts.get_accuracies "KinFaceW-II" "fs" "restricted"
	python3 -m src.scripts.get_accuracies "KinFaceW-II" "fd" "restricted"
	python3 -m src.scripts.get_accuracies "KinFaceW-II" "ms" "restricted"
	python3 -m src.scripts.get_accuracies "KinFaceW-II" "md" "restricted"

runPredictionTSK:
	python3 -m src.scripts.get_accuracies "TSKinFace" "fs" "null"
	python3 -m src.scripts.get_accuracies "TSKinFace" "fd" "null"
	python3 -m src.scripts.get_accuracies "TSKinFace" "ms" "null"
	python3 -m src.scripts.get_accuracies "TSKinFace" "md" "null"
	python3 -m src.scripts.get_accuracies "TSKinFace" "fms" "null"
	python3 -m src.scripts.get_accuracies "TSKinFace" "fmd" "null"

runPrediction:
	make runPredictionKFW1Unrestricted
	make runPredictionKFW1Restricted
	make runPredictionKFW2Unrestricted
	make runPredictionKFW2Restricted
	make runPredictionTSK

runPairwise:
	python3 -m src.scripts.get_pairwise_accuracies "KinFaceW-I" "restricted"
	python3 -m src.scripts.get_pairwise_accuracies "KinFaceW-I" "unrestricted"
	python3 -m src.scripts.get_pairwise_accuracies "KinFaceW-II" "restricted"
	python3 -m src.scripts.get_pairwise_accuracies "KinFaceW-II" "unrestricted"
	python3 -m src.scripts.get_pairwise_accuracies "TSKinFace" "null"

run_KFWI_CFN_Unrestricted:
	python3 -m src.scripts.run_WGEML "KinFaceW-I" "fs" "unrestricted" "VGG"
	python3 -m src.scripts.run_WGEML "KinFaceW-I" "fd" "unrestricted" "VGG"
	python3 -m src.scripts.run_WGEML "KinFaceW-I" "ms" "unrestricted" "VGG"
	python3 -m src.scripts.run_WGEML "KinFaceW-I" "md" "unrestricted" "VGG"

	python3 -m src.scripts.get_accuracies "KinFaceW-I" "fs" "unrestricted" "VGG"
	python3 -m src.scripts.get_accuracies "KinFaceW-I" "fd" "unrestricted" "VGG"
	python3 -m src.scripts.get_accuracies "KinFaceW-I" "ms" "unrestricted" "VGG"
	python3 -m src.scripts.get_accuracies "KinFaceW-I" "md" "unrestricted" "VGG"

run_KFWI_CFN_Restricted:
	python3 -m src.scripts.run_WGEML "KinFaceW-I" "fs" "restricted" "VGG"
	python3 -m src.scripts.run_WGEML "KinFaceW-I" "fd" "restricted" "VGG"
	python3 -m src.scripts.run_WGEML "KinFaceW-I" "ms" "restricted" "VGG"
	python3 -m src.scripts.run_WGEML "KinFaceW-I" "md" "restricted" "VGG"

	python3 -m src.scripts.get_accuracies "KinFaceW-I" "fs" "restricted" "VGG"
	python3 -m src.scripts.get_accuracies "KinFaceW-I" "fd" "restricted" "VGG"
	python3 -m src.scripts.get_accuracies "KinFaceW-I" "ms" "restricted" "VGG"
	python3 -m src.scripts.get_accuracies "KinFaceW-I" "md" "restricted" "VGG"


# E2E make target

runEndToEnd:
	make getAllFaceDescriptors
	make runWGEML
	make runPrediction

runAblationStudy:
	python3 -m src.scripts.ablation_study "KinFaceW-I" "restricted"
	python3 -m src.scripts.ablation_study "KinFaceW-I" "unrestricted"
	python3 -m src.scripts.ablation_study "KinFaceW-II" "restricted"
	python3 -m src.scripts.ablation_study "KinFaceW-II" "unrestricted"
	python3 -m src.scripts.ablation_study "TSKinFace" "null"

clean:
	find -iname "*.pyc" -delete
	find -iname "__pycache__" -delete
