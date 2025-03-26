import numpy as np

from .tool_functions import to_complex, to_real

def kernel(z, x):
    """
    Compute the Poisson kernel
    :param z: Input points
    :param x: Boundary points
    :return: Poisson kernel
    """
    w = z.reshape(-1, 1) / x.reshape(1, -1)
    w = np.real((1 + w) / (1 - w))
    return w


def integral(
    z: np.ndarray[np.floating],
    x: np.ndarray[np.complexfloating],
    y: np.ndarray[np.complexfloating],
) -> np.ndarray[np.floating]:
    """
    Compute the Poisson integral
    :param z: Internal points to be integrated, n x 2 real array
    :param x: Boundary points, m x 1 complex array
    :param y: Boundary values, m x 1 complex array
    :return: Poisson integral result
    """
    assert z.ndim == 2 and z.shape[1] == 2 and np.issubdtype(z.dtype, np.floating), (
        "z must be n x 2 array with float type"
    )

    assert (
        x.ndim == y.ndim == 1
        and x.shape == y.shape
        and np.issubdtype(x.dtype, np.complexfloating)
        and np.issubdtype(y.dtype, np.complexfloating)
    ), "x and y must be m x 1 array with complex type"

    z = to_complex(z)
    theta = np.angle(x)
    dth = np.diff(theta)
    dth = np.append(dth, theta[0] - theta[-1])
    dth[dth < 0] += 2 * np.pi
    kernel_values = kernel(z, x)
    w = (kernel_values @ (y * dth)) / (2 * np.pi)
    w = np.concatenate([y, w])
    w = to_real(w)
    return w
