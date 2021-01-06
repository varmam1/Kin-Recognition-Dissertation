import numpy as np
import sys
from ..data_preparation import save_and_load, prep_cross_valid, properly_formatted_inputs

# Usage : python3 -m src.scripts.run_WGEML [dataset] [relationship_2_char] [restricted]
# Ex. python3 -m src.scripts.run_WGEML KinFaceW-I fd unrestricted
# The above runs WGEML for the dataset KinFaceW-I with the father-daughter relationship
# And with the image-unrestricted setting in which case negative pairs are given to WGEML

dataset = sys.argv[1]
relationship = sys.argv[2]
restricted = sys.argv[3]
