from . import *
import numpy as np
from sklearn.neighbors import NearestNeighbors


def get_all_values_for_a_relationship(posPairSet, negPairSet):
    """
    Given the positive pair set and negative pair set returns a tuple
    (U, w) where U is an array of the transformation matrices for each view
    and w is an array of the combination weights for each view, where a view
    is a feature descriptor of the face.

    Keyword Arguments:
    - posPairSet: A M-tuple where M is the number of feature descriptors that
    are used. Each element of the tuple is a 2-tuple of numpy arrays of shape
    (N, size) where N is the number of samples used to train and size is the
    size of that specific type of descriptor. If this tuple is (x, y), then
    for all i = 1, ..., N, we have that x[i] and y[i] form a positive pair
    for this relationship.

    - negPairSet: The same as posPairSet except x[i] and y[i] don't have the
    specified kinship relationship.

    Returns:
    - U: The transformation matrix for the relationship
    - w: The combination weights for the relationship
    """

    # For each view, p:
    for view in range(len(posPairSet)):

        # Search the K-nearest neighbors of x_i^p and y_i^p with Euclidean distance for i = 1, ..., N
        pos_x_view, pos_y_view = posPairSet[view]
        neg_x_view, neg_y_view = negPairSet[view]

        x_nbrs = NearestNeighbors(n_neighbors=K).fit(pos_x_view)
        _, x_indices = x_nbrs.kneighbors(pos_x_view)
        y_nbrs = NearestNeighbors(n_neighbors=K).fit(pos_y_view)
        _, y_indices = x_nbrs.kneighbors(pos_y_view)

        # Construct the matrices S_p, D_p, D_{1p}, D_{2p} using the nearest neighbors
        diff_S_p = pos_x_view - pos_y_view
        S_p = np.dot(np.transpose(diff_S_p), diff_S_p)
        S_p = 1.0/(pos_x_view.shape[0])*S_p
        
        diff_D_p = neg_x_view - neg_y_view
        D_p = np.dot(np.transpose(diff_D_p), diff_D_p)
        D_p = 1.0/(neg_x_view.shape[0])*D_p
        # Modify S_p in a way so it isn't near singular
        # Get U_p using all the matrices obtained

    # Obtain w using U = [U_1, ..., U_M] for all views

