import numpy as np


def predict_if_kin_1(w, U, xs, ys, theta, triRel=False):
    """
    Given the weights and U matrices for each view along with all of the pairs
    that need to be predicted whether they are of the kin relationship being
    tested or not, returns a boolean vector corresponding to whether they are
    kin or not.

    Keyword Arguments:
    - w: The w vector of shape (M, ) where M is the amount of face descriptors
        used
    - U: The U_p matrices in which U is an array of length M and each matrix
        has shape (d, D) where d << D.
    - xs: An np array of shape (N, M, D) which has all of the face descriptors
        for each image that the prediction is wanted from. For all i in [0, N)
        xs[i] is paired with ys[i] and whether they are of the kinship relation
        is what is being tested.
    - ys: An np array of shape (N, M, D) with the same description as xs.
    - theta: A float which represents the threshold that cosine similarity
        should be if they are of the kin relation.
    - triRel: A boolean which represents whether this is being tested for a tri
        kin relationship.

    Returns:
    - A boolean np array of shape (N, ) in which each element corresponds to
        whether the corresponding pair is of the relation or not.
    """
    scoreVector = np.zeros(xs.shape[0])
    for p in range(len(w)):
        xs_p = xs[:, p]
        ys_p = ys[:, p]
        A_p = np.dot(U[p], np.transpose(U[p]))
        w_p = w[p]
        # This should be a vector of length N
        inner_product_vals = np.einsum("ij, ij->i", np.dot(xs_p, A_p), ys_p)
        x_norms = np.sqrt(np.einsum("ij, ij->i", np.dot(xs_p, A_p), xs_p))
        y_norms = np.sqrt(np.einsum("ij, ij->i", np.dot(ys_p, A_p), ys_p))

        cosine_vals = np.divide(inner_product_vals, np.multiply(x_norms, y_norms))
        similarlity_vals = w_p * (cosine_vals + 1)/2
        scoreVector = scoreVector + similarlity_vals

    if triRel:
        # If tri-kin relationship, mean similarity between father and mother scores
        fatherScore = scoreVector[:int(len(scoreVector)/2)]
        motherScore = scoreVector[int(len(scoreVector)/2):]
        scoreVector = (fatherScore + motherScore)/2
    return scoreVector >= theta
