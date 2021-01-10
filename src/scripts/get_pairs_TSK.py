from scipy.io import savemat
import numpy as np
import os
from .. import dataPath

fd_pairs = []
fs_pairs = []
md_pairs = []
ms_pairs = []
fms_pairs = []
fmd_pairs = []

# FMD
images = os.listdir(dataPath + "TSKinFace/images/FMD")
for i in range(len(images)//3):
    fd_pairs.append([images[3*i + 1], images[3*i]])
    md_pairs.append([images[3*i + 2], images[3*i]])
    fmd_pairs.append(images[3*i + 1 : 3*i + 3] + [images[3*i]])


# FMS
images = os.listdir(dataPath + "TSKinFace/images/FMS")
for i in range(len(images)//3):
    fs_pairs.append([images[3*i], images[3*i + 2]])
    ms_pairs.append([images[3*i + 1], images[3*i + 2]])
    fms_pairs.append(images[3*i : 3*i + 3])


# FMSD
images = os.listdir(dataPath + "TSKinFace/images/FMSD")
for i in range(len(images)//4):
    fs_pairs.append([images[4*i + 1], images[4*i + 3]])
    ms_pairs.append([images[4*i + 2], images[4*i + 3]])
    fd_pairs.append([images[4*i + 1], images[4*i]])
    md_pairs.append([images[4*i + 2], images[4*i]])
    fmd_pairs.append(images[4*i + 1 : 4*i + 3] + [images[4*i]])
    fms_pairs.append(images[4*i + 1 : 4*i + 4])

savemat(dataPath + "TSKinFace/meta_data/fs_pairs.mat", {"pairs" : np.array(fs_pairs)})
savemat(dataPath + "TSKinFace/meta_data/fd_pairs.mat", {"pairs" : np.array(fd_pairs)})
savemat(dataPath + "TSKinFace/meta_data/md_pairs.mat", {"pairs" : np.array(md_pairs)})
savemat(dataPath + "TSKinFace/meta_data/ms_pairs.mat", {"pairs" : np.array(ms_pairs)})
savemat(dataPath + "TSKinFace/meta_data/fmd_pairs.mat", {"pairs" : np.array(fmd_pairs)})
savemat(dataPath + "TSKinFace/meta_data/fms_pairs.mat", {"pairs" : np.array(fms_pairs)})
