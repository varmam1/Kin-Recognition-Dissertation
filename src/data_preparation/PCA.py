import numpy as np
from sklearn.decomposition import PCA


def reduce_dimensions(vectors, dim=200, amount_to_truncate=200):
    """
    Given a numpy array of face descriptor vectors, reduces the dimensions
    to the inputted number and returns the array of the reduced vectors.

    Keyword Arguments:
    - vectors: A numpy array of the face descriptors that need to be reduced
    - dim (int): The amount of dimensions in the outputted vector. Should be
        less than or equal to the amount of vectors.
    - amount_to_truncate (int): The amount of the dimensions of the reduced
        vector are wanted.

    Returns:
    - A numpy array of the reduced vectors in the same element position.
    For example, the reduced vector of vectors[0] would be in the 0th element
    of this numpy array.
    """
    pca = PCA(n_components=dim)
    return pca.fit_transform(vectors)[:, :amount_to_truncate]
