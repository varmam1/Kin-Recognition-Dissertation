from . import *
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy import linalg


# TODO: Change function name
def get_diff_mat(x_view, y_view):
    """
    Returns 1/N * \sigma_{(x_i, y_i)} (x_i - y_i)^T(x_i - y_i)

    Keyword Arguements:
    - x_view: An np array of shape (N, D) where N is the number of samples
        and D is the dimension of the descriptor vector.
    - y_view: An np array of shape (N, D) where N is the number of samples
        and D is the dimension of the descriptor vector.

    Returns:
    - The matrix defined above.
    """
    N = x_view.shape[0]

    diff_mat = pos_x_view - pos_y_view
    mat = np.dot(np.transpose(diff_mat), diff_mat)
    mat = (1.0/N)*mat

    return mat


def get_penalty_graphs(pos_x_view, pos_y_view, x_neighbor_indices, y_neighbor_indices):
    """
    Given the positive pairs and the neighbors of each of the vectors, returns
    the penalty graphs D_1p and D_2p as defined in the paper.

    Keyword Arguments:
    - pos_x_view: For the given view, the vectors for one side of the correct
        pairs for the relationship being tested.
    - pos_y_view: For the given view, the vectors for the other side of the
        correct pairs for the relationship being tested.
    - x_neighbor_indices: An np array with shape (N, K) where, for each i, the
        array at index i represents the indices in pos_x_view of the K closest
        neighbors to the vector at index i in pos_x_view.
    - y_neighbor_indices: The same as x_neighbor_indices but for pos_y_view
        instead of pos_x_view.

    Returns:
    - The matrices D_1p and D_2p as described in the paper.
    """
    dim = pos_x_view.shape[1]
    N = pos_x_view.shape[0]
    D_1p = np.zeros((dim, dim))
    D_2p = np.zeros((dim, dim))

    for i in range(pos_x_view.shape[0]):
        x_neighbors = x_neighbor_indices[i]
        y_neighbors = y_neighbor_indices[i]
        x_i = pos_x_view[i]
        y_i = pos_y_view[i]

        for k in range(K):
            diff1 = np.expand_dims((x_i - y_neighbors[k]), axis=0)
            diff2 = np.expand_dims((x_neighbors[k] - y_i), axis=0)
            D_1p = D_1p + np.dot(np.transpose(diff1), diff1)
            D_2p = D_2p + np.dot(np.transpose(diff2), diff2)

    D_1p = 1.0/(N * K) * D_1p
    D_2p = 1.0/(N * K) * D_2p

    return (D_1p, D_2p)


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
    weights = np.zeros(len(posPairSet))
    transformation_matrices = []

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

        S_p = get_diff_mat(pos_x_view, pos_y_view)
        D_p = get_diff_mat(neg_x_view, neg_y_view)

        D_1p, D_2p = get_penalty_graphs(pos_x_view, pos_y_view, x_indices, y_indices)

        # Modify S_p in a way so it isn't near singular
        S_p = (1-beta) * S_p + beta*np.trace(S_p)/N * np.identity(S_p.shape[0])

        # Get U_p using all the matrices obtained
        # Use the corresponding vectors to the top d eigenvalues to make up U_p
        # This will be U_p = [u_1, u_2, ..., u_d] where lambda_1 >= lambda_2 >= ...

        # TODO: Figure out where this d comes from
        d = 10
        combination_of_all_D = 0.5*(D_1p + D_2p) + D_p
        eig_vals, eig_vecs = linalg.eig(combination_of_all_D, S_p)

        # TODO: Potentially way too expensive as it's sorting the eigenvalues fully
        U_p = np.transpose(eig_vecs[:d, eig_vals.argsort()[::-1]]) 
        U.append(U_p)

        # Obtain w_p using U_p

        w_p = np.trace(np.dot(np.dot(np.transpose(U_p), S_p), U_p))
        w_p = w_p/np.trace(np.dot(np.dot(np.transpose(U_p), combination_of_all_D), U_p))
        w_p = (w_p)**(1.0/(r - 1))
        weights[view] = w_p
    
    weights = weights/np.sum(weights)
    return (transformation_matrices, weights)
