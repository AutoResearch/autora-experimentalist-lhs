import numpy as np

from autora.experimentalist.lhs import sample


def test_output_dimensions():
    X = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    n = 2
    X_new = sample(X, n)

    # Check that the sampler returns n experiment conditions
    assert X_new.shape == (n, X.shape[1])


# Note: We encourage you to adjust this test and write more tests.
