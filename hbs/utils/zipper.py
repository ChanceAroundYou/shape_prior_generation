import numpy as np


def zipper(bound, others=None):
    if others is None:
        others = []

    n = len(bound)
    params = np.zeros(n + 1, dtype=complex)
    params[:2] = bound[:2]
    points = f_pre(np.concatenate([bound, others]), params[0], params[1])
    for j in range(2, n):
        params[j] = points[j]
        points = f(points, points[j])
    params[n] = points[0]
    points = f_end(points, params[n])
    points = f_final(points)

    bound = points[:n]
    others = points[n:]
    return bound, others, params



def f_pre(z, p, q):
    """
    map z to right half plane, where real part is positive.
    p -> inf,
    q -> 0,
    note that always p = z[0], q = z[1]
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        w = (z - q) / (z - p)
    w[np.isinf(z)] = 1
    w[z == p] = np.inf
    w = np.sqrt(w)
    return w


def f1(z, p):
    """
    0 -> 0
    p -> 1
    aj -> bj
    inf -> - c/d * 1j
    1j/d -> inf
    """
    c = np.real(p) / np.abs(p) ** 2
    d = np.imag(p) / np.abs(p) ** 2
    # if d == 0:
    #     s = z[:10]
    #     print(s, p)
    #     print(s.imag)
    #     from matplotlib import pyplot as plt
    #     plt.scatter(s.real, s.imag)
    #     plt.show()
    if d == 0:
        with np.errstate(divide="ignore", invalid="ignore"):
            w = z * c
        w[np.isinf(z)] = np.inf
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            w = c * z / (1  + 1j * d * z)
        w[np.isinf(z)] = -c / d * 1j
        w[z == 1j / d] = np.inf
        w[np.isclose(z, p)] = 1
    return w


def f2(z):
    """
    1 -> 0
    0 -> 1j
    bj -> sqrt(b^2 + 1)j (positive imaginary part)
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        w = np.sqrt(z**2 - 1)
    k = np.imag(w) * np.imag(z) < 0
    w[k] = -w[k]
    w[z == 0] = 1j
    w[np.isinf(z)] = np.inf
    return w


def f(z, p):
    """
    map p to 0,
    with keep points in y-axis, going up.
    0 -> 0 -> 1j
    p -> 1 -> 0
    aj -> bj -> sqrt(b^2 + 1)j
    """
    if p.real == 0:
        raise ValueError("Point p should not be a pure imaginary number.")

    w = f1(z, p)
    w = f2(w)
    w[np.isclose(z, p)] = 0
    return w


def f_end(z, p):
    q = 1 - z / p
    with np.errstate(divide="ignore", invalid="ignore"):
        w = (z / q) ** 2
    w[np.isinf(z)] = p**2
    w[q == 0] = np.inf
    return w


def f_final(z):
    with np.errstate(divide="ignore", invalid="ignore"):
        w = (z - 1j) / (z + 1j)
    w[np.isinf(z)] = 1
    w[z == -1j] = np.inf
    return w


def zipper_params(points, params):
    n = len(params) - 1
    points = f_pre(points, params[0], params[1])
    for j in range(2, n):
        points = f(points, params[j])
    points = f_end(points, params[n])
    points = f_final(points)
    return points


def zipper_inv(points, params):
    n = len(params) - 1
    points = f_final_inv(points)
    points = f_end_inv(points, params[n])
    for j in range(n - 1, 1, -1):
        points = f_inv(points, params[j])
    points = f_pre_inv(points, params[0], params[1])
    return points


def f_pre_inv(w, p, q):
    z = (p * w**2 - q) / (w**2 - 1)
    z[np.isinf(w)] = p
    z[w == 1] = np.inf
    return z


def f1_inv(w, p):
    pc = np.real(p) / np.abs(p) ** 2
    pd = np.imag(p) / np.abs(p) ** 2
    z = w / (pc - 1j * pd * w)
    z[np.isinf(w)] = 1j / pd
    z[w == -1j * pc / pd] = np.inf
    return z


def f2_inv(w):
    z = np.sqrt(w**2 + 1)
    k = np.imag(w) * np.imag(z) < 0
    z[k] = -z[k]
    z[w == 1j] = 0
    z[np.isinf(w)] = np.inf
    return z


def f_inv(w, p):
    z = f2_inv(w)
    z = f1_inv(z, p)
    return z


def f_end_inv(w, p):
    z = np.sqrt(w)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = z / (1 + z / p)
    z[np.isinf(w)] = p
    return z


def f_final_inv(w):
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (w + 1) * 1j / (1 - w)
    k = np.abs(np.imag(z)) < 1e-10
    z[k] = np.real(z[k])
    z[np.isinf(w)] = -1j
    z[w == 1] = np.inf
    return z


# def zipper_d(points, params):
#     n = len(params) - 1
#     d = f_pre_d(points, params[0], params[1])
#     points = f_pre(points, params[0], params[1])
#     for j in range(2, n):
#         d *= f_d(points, params[j])
#         points = f(points, params[j])
#     d *= f_end_d(points, params[n])
#     points = f_end(points, params[n])
#     d *= f_final_d(points)
#     return d

# def f_pre_d(z, p, q):
#     t = f_pre(z, p, q)
#     d = (q - p) / (2 * ((z - p) ** 2) * t)
#     d[np.isinf(z)] = 0
#     d[z == p] = np.inf
#     d[z == q] = np.inf
#     return d


# def f1_d(z, p):
#     pc = np.real(p) / np.abs(p) ** 2
#     pd = np.imag(p) / np.abs(p) ** 2
#     w = pc / (1 + 1j * pd * z) ** 2
#     w[np.isinf(z)] = 0
#     w[z == 1j / pd] = np.inf
#     return w


# def f2_d(z):
#     d = z / f2(z)
#     d[np.isinf(z)] = 1
#     return d


# def f_d(z, p):
#     t = f1(z, p)
#     d1 = f1_d(z, p)
#     d2 = f2_d(t)
#     d = d1 * d2
#     return d


# def f_end_d(z, p):
#     d = 2 * z / (1 - z / p) ** 3
#     d[np.isinf(z)] = 0
#     d[z == p] = np.inf
#     return d


# def f_final_d(z):
#     d = 2j / (z + 1j) ** 2
#     d[np.isinf(z)] = 0
#     d[z == -1j] = np.inf
#     return d

# def zipper_inv_d(points, params):
#     n = len(params) - 1
#     dl = np.zeros(n + 1, dtype=complex)
#     d = f_final_inv_d(points)
#     dl[n] = np.angle(d)
#     points = f_final_inv(points)
#     d0 = f_end_inv_d(points, params[n])
#     d *= d0
#     dl[n - 1] = np.angle(d)
#     points = f_end_inv(points, params[n])
#     for j in range(n - 1, 1, -1):
#         d0 = f_inv_d(points, params[j])
#         d *= d0
#         dl[j - 1] = np.angle(d)
#         points = f_inv(points, params[j])
#     d0 = f_pre_inv_d(points, params[0], params[1])
#     dl[0] = np.angle(d)
#     d *= d0
#     return d

# def f_pre_inv_d(w, p, q):
#     d = (q - p) * 2 * w / (w**2 - 1) ** 2
#     d[w**2 == 1] = np.inf
#     d[np.isinf(w)] = 0
#     return d


# def f1_inv_d(w, p):
#     pc = np.real(p) / np.abs(p) ** 2
#     pd = np.imag(p) / np.abs(p) ** 2
#     d = pc / (pc - 1j * pd * w) ** 2
#     d[np.isinf(w)] = 0
#     d[w == 1j * pc / pd] = np.inf
#     return d


# def f2_inv_d(w):
#     d = w / f2_inv(w)
#     d[np.isinf(w)] = 1
#     return d


# def f_inv_d(w, p):
#     t = f2_inv(w)
#     d1 = f1_inv_d(t, p)
#     d2 = f2_inv_d(w)
#     d = d1 * d2
#     return d


# def f_end_inv_d(w, p):
#     q = np.sqrt(w)
#     d = 1 / (2 * q * (1 + q / p) ** 2)
#     d[np.isinf(w)] = 0
#     d[q == 0] = np.inf
#     d[q == -p] = np.inf
#     return d


# def f_final_inv_d(w):
#     d = 2j / (1 - w) ** 2
#     d[np.isinf(w)] = -2j
#     d[w == 1] = np.inf
#     return d
