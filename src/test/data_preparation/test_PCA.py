import pytest
import numpy as np
from ...data_preparation import PCA as reduction
from sklearn.decomposition import PCA

def test_reduce_dimensions_with_PCA():
    vectors = np.array([[1, 2,  3, 4],
                        [9, 3,  5, 1],
                        [4, 6, -1, 3],
                        [6, 2, -2, 4]])
    testPCA = PCA(n_components=2)
    testPCA.fit(vectors)
    expectedOut = testPCA.transform(vectors)[:, :1]
    out = reduction.reduce_dimensions(vectors, dim=2, amount_to_truncate=1)
    assert np.isclose(out, expectedOut).all()
