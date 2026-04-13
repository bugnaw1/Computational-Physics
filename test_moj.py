import numpy as np
from moj import compute_potential

def test_shape_preserved():
    p = np.zeros((10, 10))
    result = compute_potential(p, 10)

    assert result.shape == (10, 10)

def test_boundaries_unchanged():
    p = np.zeros((10, 10))

    p[0, :] = 1
    p[-1, :] = 2
    p[:, 0] = 3
    p[:, -1] = 4

    original = p.copy()

    result = compute_potential(p, 50)

    assert np.allclose(result[0, :], original[0, :])
    assert np.allclose(result[-1, :], original[-1, :])
    assert np.allclose(result[:, 0], original[:, 0])
    assert np.allclose(result[:, -1], original[:, -1])

def test_interior_changes():
    p = np.zeros((10, 10))
    p[0, :] = 1
    p[-1, :] = 1
    p[:, 0] = 1
    p[:, -1] = 1

    result = compute_potential(p.copy(), 10)

    assert not np.allclose(result, p)