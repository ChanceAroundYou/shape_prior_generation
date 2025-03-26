import numpy as np


def mu_chop(mu, bound=0.9999, constant=None):
    """
    Perform truncation on mu
    :param mu: Input complex array
    :param bound: Truncation boundary value
    :param constant: Constant value after truncation
    :return: Processed mu
    """
    if constant is None:
        constant = bound

    abs_mu = np.abs(mu)
    idx = abs_mu >= bound
    mu[idx] = constant * (mu[idx] / abs_mu[idx])
    return mu


def to_real(x: np.ndarray[np.complexfloating]) -> np.ndarray[np.floating]:
    assert isinstance(x, np.ndarray) and np.issubdtype(x.dtype, np.complexfloating), (
        "x must be a complex array"
    )
    assert x.ndim == 1, "x array must be 1D array"
    return np.stack([x.real, x.imag], axis=-1)


def to_complex(x: np.ndarray[np.floating]) -> np.ndarray[np.complexfloating]:
    # assert , "Input must be a numpy array"
    # assert np.issubdtype(x.dtype, np.floating) or np.issubdtype(x.dtype, np.integer), (
    #     "Input array must have floating point or integer data type"
    # )
    assert isinstance(x, np.ndarray) and np.issubdtype(x.dtype, np.floating), (
        "x must be a real array"
    )
    assert x.ndim == 2 and x.shape[1] == 2, "x must be n x 2 array"
    return x[:, 0] + 1j * x[:, 1]
