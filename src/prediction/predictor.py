import numpy as np

def get_similarity(w, U, xs, ys, triRel=False, fds_included = None):
    """
    Given the weights and U matrices for each view along with all of the pairs
    that need to be predicted whether they are of the kin relationship being
    tested or not, returns a vector corresponding to the similarity of the pair.

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
    - An np array of shape (N, ) in which each element corresponds to
        the similarity score of the corresponding pair.

    """
    if fds_included is None:
        fds_included = np.ones(w.shape[0], dtype=bool)

    scoreVector = np.zeros(xs.shape[0])
    weights = np.zeros(w.shape[0])
    for i in range(len(w)):
        if fds_included[i]:
            weights[i] = w[i]
    
    weights = weights / np.sum(weights)
    for p in range(len(w)):
        if fds_included[p]:
            xs_p = xs[:, p]
            ys_p = ys[:, p]
            A_p = np.dot(U[p], np.transpose(U[p]))
            w_p = weights[p]
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
    return scoreVector


def predict(w, U, xs, ys, theta, triRel=False, fds_included = None):
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
    
    return get_similarity(w, U, xs, ys, triRel=triRel, fds_included=fds_included) >= theta
