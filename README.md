# Kin Recognition using Weighted Graph Embedding-Based Metric Learning


## General Overview of the Project

The project implements the paper Weighted Graph Embedding-Based Metric Learning for Kinship Verification (Liang et al., 2019) for the purposes of kinship verification and tri-kinship verification using Python 3.6. The algorithm, WGEML, is implemented and trained on the KinFaceW-I, KinFaceW-II, and TSKinFace datasets. Each of the KinFaceW datasets have 2 settings, unrestricted and restricted, in which unrestricted means that both positive and negative pairs are used in the training process whereas restricted only uses positive pairs. A model is created for a specified dataset, setting and relationship, such as father-daughter (FD). 

## Original Aims of the Project
Kin recognition is the ability to recognize whether two people are related to each other just by looking at them. The goal of the project was to implement a state-of-the-art model, Weighted Graph Embedding-Based Metric Learning (WGEML), to solve the Kin Verification problem and to verify the accuracies obtained. I also wanted to further evaluate the models obtained by looking at potential biases in the datasets used in the paper.

## Work Completed
I successfully implemented WGEML and was able to recreate the original accuracies within an acceptable range. Furthermore, I performed ablation studies in order to determine if there is a relationship between the number of face descriptors used and the accuracy of the model. I also replaced one face descriptor with a less computationally expensive one. I explored biases in the datasets used using the model that was created and discussed the implications of such biases on the results. 

## Setup

Firstly, the dependencies must be obtained which can be done running:

```bash
make init
```

in the root folder of the project. To make sure everything is working properly, the unit tests can be ran using:

```bash
make coverage
```

which should come back all fine. 

## Preprocessing

To compute the face descriptors for each of the datasets, the make command:

```bash
make getAllFaceDescriptors
```

can be run. NB: This takes a while to run and the VGG network is used to compute face descriptors so it might be useful to run this on a GPU. 

Furthermore, the cross validation splits for TSKinFace along with negative pairs need to be generated, or the ones that are in the repo can be used if wanted. If you want to get new splits, the following can be run:

```bash
python3 -m src.scripts.preprocessing_TSK
```

## Training

To train the model for a given dataset, setting, and relationship:

```bash
python3 -m src.scripts.training [dataset] [2-character relationship code] [restricted/unrestricted]
```

Should be run. To run this with TSKinFace which has no unrestricted vs restricted setting, just set the third parameter to be "null". This can all be ran using the make commands such as:

```bash
make runWGEMLKFW1Unrestricted
```

Which trains WGEML on KinFaceW-I Unrestricted for each relationship. To train each dataset, setting and relationship, run:

```bash
make runWGEML
```

## Testing

Similarly for testing, for a given dataset, setting and relationship:

```bash
python3 -m src.scripts.testing [dataset] [2-character relationship code] [restricted/unrestricted]
```

Should be run which prints out the accuracies for each individual fold in the cross-validation splits and the average accuracy among them. Similarly to testing, to run it for a dataset and setting, we can do something like:

```bash
make runPredictionKFW1Unrestricted
```

And we can run:

```bash
make runPrediction
```

To print out the accuracies for each dataset, setting and relationship supported. 

## Running End-to-End

Although it is suggested to run each part individually, a make target is created to run it end to end which is:

```bash
make runEndToEnd
```

