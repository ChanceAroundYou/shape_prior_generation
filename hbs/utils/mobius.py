import numpy as np

def mobius(z, a, theta=None):
    if theta is None:
        k = 1
    else:
        k = np.exp(theta * 1j)

    if a != 0:
        w = k * (z - a) / (1 - np.conj(a) * z)
        w[np.isinf(z)] = -k / np.conj(a)
        w[np.isclose(z, 1 / np.conj(a), atol=1e-15)] = np.inf
    else:
        w = k * z
    return w


def mobius_inv(z, a, theta=None):
    if theta is None:
        return mobius(z, -a)
    else:
        k = np.exp(theta * 1j)
    return mobius(z, -k * a, -theta)


def mobius_d(z, a, theta):
    return np.exp(theta * 1j) * (1 - abs(a) ** 2) / (1 - np.conj(a) * z) ** 2