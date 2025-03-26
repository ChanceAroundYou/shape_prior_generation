import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

from .utils.mobius import mobius, mobius_inv
from .utils.tool_functions import to_complex
from .utils.zipper import zipper, zipper_params


class ConformalWelding:
    def __init__(
        self,
        x: np.ndarray[np.complexfloating],
        y: np.ndarray[np.complexfloating],
        params: np.ndarray[np.complexfloating],
    ):
        """
        Initialize the ConformalWelding class.

        Parameters:
        x (np.ndarray): Complex ndarray of shape (n,) representing initial x points.
        y (np.ndarray): Complex ndarray of shape (n,) representing initial y points.
        params (np.ndarray): Complex ndarray of shape (n+1,) representing parameters.
        """
        assert (
            x.ndim == y.ndim == params.ndim == 1
            and np.issubdtype(x.dtype, np.complexfloating)
            and np.issubdtype(y.dtype, np.complexfloating)
            and np.issubdtype(params.dtype, np.complexfloating)
        ), "Input must be 1D complex array"
        assert len(x) == len(y) == len(params) - 1, (
            "Length of x, y, params must be n, n, n+1"
        )

        self.params = params
        self._set_init_x(x)
        self._set_init_y(y)

    def _set_init_x(self, x: np.ndarray[np.complexfloating]):
        x_angle = np.angle(x)
        x_angle_diff = np.diff(x_angle)
        min_pos = np.argmin(x_angle_diff)
        if x_angle_diff[min_pos] < 0:
            x_angle[min_pos + 1 :] += 2 * np.pi
        # x_angle -= x_angle[0]
        # x_angle = np.mod(x_angle, 2 * np.pi)
        x = np.exp(x_angle * 1j)

        self.init_x_angle = x_angle
        self.init_x = x
        self.x = x

    def _set_init_y(self, y: np.ndarray[np.complexfloating]):
        self.init_y = y
        self.y = y

    def uniquify(self):
        x_angle, idx = np.unique(np.angle(self.init_x), return_index=True)
        x = np.exp(x_angle * 1j)
        y = self.init_y[idx]
        y = np.exp(np.angle(y) * 1j)

        self._set_init_x(x)
        self._set_init_y(y)

    def linear_interp(self, num: int):
        x_angle_regular = np.arange(0, 2 * np.pi, 2 * np.pi / num)
        self.x = np.exp(x_angle_regular * 1j)

        inter_func = interp1d(
            np.concatenate(
                [
                    self.init_x_angle - 2 * np.pi,
                    self.init_x_angle,
                    self.init_x_angle + 2 * np.pi,
                ]
            ),
            np.concatenate([self.init_y, self.init_y, self.init_y]),
            kind="linear",
        )
        y = inter_func(x_angle_regular)
        y_anlge = np.angle(y)
        self.y = np.exp((y_anlge - y_anlge[0]) * 1j)

    def rotate_x(self, r):
        x_angle = np.mod(self.init_x_angle - r, 2 * np.pi)
        x = np.exp(x_angle * 1j)
        self._set_init_x(x)

    def x_post_norm(self):
        """
        对外部点进行后归一化
        :param z: 输入点
        :param params: 参数
        :return: 归一化后的点，a 和 theta 参数
        """
        zinf = zipper_params([np.inf], self.params)
        a = -1 / np.conj(zinf)
        theta = 0
        x = mobius_inv(self.x, a, theta)
        x = np.flipud(x)
        self._set_init_x(x)

    def y_post_norm(self, eps=1e-10):
        """
        对内部点进行后归一化
        :param z: 输入点
        :return: 归一化后的点，a 和 theta 参数
        """
        p = np.append(self.y, [0, 1])
        while True:
            center = np.mean(p[:-2])
            if np.abs(center) <= np.finfo(np.float64).eps:
                break
            p = mobius(p, center)

        y = p[:-2]
        y_angle = np.angle(y)
        y_angle_diff = np.diff(np.insert(y_angle, -1, y_angle[0]))
        y_angle_diff[y_angle_diff < -np.pi] += 2 * np.pi
        y_angle_diff[(-0.1 < y_angle_diff) & (y_angle_diff < 0)] *= -1
        # y_angle_diff[y_angle_diff == 0] = y_angle_diff[y_angle_diff > 0].min()
        y_angle_diff[y_angle_diff < eps] = eps

        if y_angle_diff.sum() > 2 * np.pi:
            y_angle_diff *= 2 * np.pi / y_angle_diff.sum()

        y_angle_new = np.cumsum(y_angle_diff)
        y_angle_new = np.insert(y_angle_new[:-1] + y_angle[0], 0, y_angle[0])
        y_new = np.exp(y_angle_new * 1j)
        # y_angle = self.y_angle_norm()
        # y = np.exp(y_angle * 1j)
        self._set_init_y(y_new)

    def plot_x(self):
        plt.gca().set_aspect("equal", adjustable="box")
        plt.scatter(self.x.real, self.x.imag)
        plt.show()

    def plot_y(self):
        plt.gca().set_aspect("equal", adjustable="box")
        plt.scatter(self.y.real, self.y.imag)
        plt.show()

    def plot(self, is_interp=True):
        plt.gca().set_aspect("equal", adjustable="box")
        x_angle = np.angle(self.x)
        x_angle = np.mod(x_angle, 2 * np.pi)
        y_angle = self.get_y_angle()

        if is_interp:
            plt.plot(x_angle, y_angle, linestyle="-", linewidth=2)
        else:
            plt.scatter(x_angle, y_angle, s=2)
        plt.show()

    def get_y_angle_diff(self):
        y_angle = np.angle(self.y)
        y_angle = np.insert(y_angle, -1, y_angle[0])
        y_angle_diff = np.diff(y_angle)

        # plt.plot(y_angle)
        # plt.plot(y_angle_diff)
        # plt.show()

        # y_angle_diff = np.insert(y_angle_diff, 0, 0)
        y_angle_diff[y_angle_diff < -np.pi] += 2 * np.pi
        y_angle_diff[(-0.1 < y_angle_diff) & (y_angle_diff < 0)] *= -1
        y_angle_diff[y_angle_diff == 0] = y_angle_diff[y_angle_diff > 0].min()

        if np.any(y_angle_diff < 0):
            raise ValueError("y_angle_diff must be non-negative")

        if y_angle_diff.sum() > 2 * np.pi:
            y_angle_diff *= 2 * np.pi / y_angle_diff.sum()

        return y_angle_diff

    def get_y_angle(self):
        y_angle_diff = self.get_y_angle_diff()
        # y_angle_diff = np.insert(y_angle_diff, 0, 0)[:-1]
        y_angle = np.cumsum(y_angle_diff)[:-1]
        y_angle = np.insert(y_angle, 0, 0)
        return y_angle


def get_conformal_welding(bound: np.ndarray[np.floating]) -> ConformalWelding:
    assert (
        bound.ndim == 2
        and bound.shape[1] == 2
        and np.issubdtype(bound.dtype, np.floating)
    ), "bound must be n x 2 real array with float type"

    bound = to_complex(bound)
    x, _, x_params = zipper(bound)
    # x = np.flipud(x)
    y, _, _ = zipper(np.flipud(bound))

    cw = ConformalWelding(x, y, x_params)
    cw.x_post_norm()
    cw.y_post_norm()
    cw.uniquify()
    return cw
