from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial import Delaunay


class Mesh:
    def __init__(self, vert: np.ndarray, face: np.ndarray | None = None):
        assert vert.ndim == 2
        assert vert.shape[1] == 2
        self.vert = vert

        if face is None:
            self.face = Delaunay(vert).simplices

        else:
            assert face.shape[1] == 3
            assert face.ndim == 2
            self.face = face

        self.vert_num = self.vert.shape[0]
        self.face_num = self.face.shape[0]

        self.face_center = self.get_face_center()
        self.edges = self.get_edge()
        self.area = self.get_area()
        self.set_operator()

    def get_face_center(self):
        return np.mean(self.vert[self.face], axis=1)

    def get_edge(self):
        e1 = self.vert[self.face[:, 2]] - self.vert[self.face[:, 1]]
        e2 = self.vert[self.face[:, 0]] - self.vert[self.face[:, 2]]
        e3 = self.vert[self.face[:, 1]] - self.vert[self.face[:, 0]]
        return e1, e2, e3

    def get_area(self):
        return np.cross(self.edges[0], self.edges[1]) / 2

    def set_operator(self):
        def get_diff():
            Mi = np.repeat(np.arange(self.face_num), 3)
            Mj = self.face.flatten()

            stack_area = np.vstack([self.area, self.area, self.area])

            M = np.stack(self.edges, axis=0)
            Mx = M[:, :, 1] / (2 * stack_area)
            My = -M[:, :, 0] / (2 * stack_area)
            Mx = Mx.T.flatten()
            My = My.T.flatten()

            Dx = csr_matrix((Mx, (Mi, Mj)))
            Dy = csr_matrix((My, (Mi, Mj)))
            Dz = (Dx - 1j * Dy) / 2
            Dc = (Dx + 1j * Dy) / 2
            return Dx, Dy, Dz, Dc

        def f2v():
            ring = defaultdict(list)
            for i, face in enumerate(self.face):
                for vertex in face:
                    ring[vertex].append(i)
            ring = [ring[i] for i in range(self.vert_num)]

            avg = np.array([len(r) for r in ring])

            A = csr_matrix(
                (1 / avg, (np.arange(self.vert_num), np.arange(self.vert_num)))
            )

            x_pos = np.concatenate([np.full(len(r), i) for i, r in enumerate(ring)])
            y_pos = np.concatenate(ring)
            B = csr_matrix(
                (np.ones(len(y_pos)), (x_pos, y_pos)),
                shape=(self.vert_num, self.face_num),
            )
            matrix = A @ B
            return matrix

        def v2f():
            x_pos = np.repeat(np.arange(self.face_num), 3)
            y_pos = self.face.flatten()
            matrix = csr_matrix(
                (np.ones(len(y_pos)) / 3, (x_pos, y_pos)),
                shape=(self.face_num, self.vert_num),
            )
            return matrix

        def laplacian():
            f1, f2, f3 = self.face.T
            e1, e2, e3 = self.edges

            l1 = (e1**2).sum(axis=1)
            l2 = (e2**2).sum(axis=1)
            l3 = (e3**2).sum(axis=1)

            cot12 = (l1 + l2 - l3) / (2 * self.area)
            cot23 = (l2 + l3 - l1) / (2 * self.area)
            cot31 = (l1 + l3 - l2) / (2 * self.area)
            diag1 = -cot12 - cot31
            diag2 = -cot12 - cot23
            diag3 = -cot31 - cot23

            x_pos = np.concatenate([f1, f2, f2, f3, f3, f1, f1, f2, f3])
            y_pos = np.concatenate([f2, f1, f3, f2, f1, f3, f1, f2, f3])
            value = np.concatenate(
                [cot12, cot12, cot23, cot23, cot31, cot31, diag1, diag2, diag3]
            )
            matrix = csr_matrix(
                (value, (x_pos, y_pos)), shape=(self.vert_num, self.vert_num)
            )
            return matrix

        self.Dx, self.Dy, self.Dz, self.Dc = get_diff()
        self.f2v = f2v()
        self.v2f = v2f()
        self.laplacian = laplacian()

    def plot(self, with_face_center: bool = False):
        # plt.figure(figsize=(8, 8))
        plt.gca().set_aspect("equal", adjustable="box")
        plt.triplot(self.vert[:, 0], self.vert[:, 1], self.face)
        if with_face_center:
            plt.plot(self.face_center[:, 0], self.face_center[:, 1], "o")
        plt.show()

    def plot_mu(self, mu: np.ndarray[np.complexfloating], is_3d: bool = False):
        assert isinstance(mu, np.ndarray) and np.issubdtype(
            mu.dtype, np.complexfloating
        ), "mu must be complex array"
        assert mu.ndim == 1 and mu.shape[0] == self.face_num, (
            "mu must be 1D array with length equal to face number"
        )

        magnitude = np.abs(mu)
        magnitude[magnitude > 1] = 1
        # 绘制散点图
        if not is_3d:
            plt.scatter(
                self.face_center[:, 0],
                self.face_center[:, 1],
                c=magnitude,
                cmap="jet",
                s=0.5,
            )
            plt.colorbar(label="Magnitude")
        else:
            ax = plt.axes(projection="3d")
            ax.scatter(
                self.face_center[:, 0],
                self.face_center[:, 1],
                magnitude,
                c=magnitude,
                cmap="jet",
                s=0.5,
            )
        plt.show()

    def plot_mapping(self, mapping: np.ndarray[np.floating], is_3d: bool = False):
        assert isinstance(mapping, np.ndarray) and np.issubdtype(
            mapping.dtype, np.floating
        ), "mapping must be real float array"
        assert mapping.ndim == 2 and mapping.shape == (self.vert_num, 2), (
            "mapping must be n x 2 array"
        )

        magnitude = np.linalg.norm(mapping, axis=1)
        # magnitude[magnitude > 1] = 1
        if not is_3d:
            plt.scatter(
                self.vert[:, 0],
                self.vert[:, 1],
                c=magnitude,
                cmap="jet",
                s=0.5,
            )
            plt.colorbar(label="Magnitude")
        else:
            ax = plt.axes(projection="3d")
            ax.scatter(
                self.vert[:, 0],
                self.vert[:, 1],
                magnitude,
                c=magnitude,
                cmap="jet",
                s=0.5,
            )
        plt.show()


class DiskMesh(Mesh):
    def __init__(
        self,
        circle: np.ndarray,
        in_vert: np.ndarray | None = None,
        out_vert: np.ndarray | None = None,
        face: np.ndarray | None = None,
    ):
        assert circle.ndim == 2
        assert circle.shape[1] == 2
        self.circle = circle
        self.circle_num = self.circle.shape[0]
        vert = circle

        if in_vert is not None:
            assert in_vert.ndim == 2
            assert in_vert.shape[1] == 2

            self.in_vert = in_vert
            self.in_vert_num = self.in_vert.shape[0]
            vert = np.vstack([vert, in_vert])
        else:
            self.in_vert = None
            self.in_vert_num = 0

        if out_vert is not None:
            assert out_vert.ndim == 2
            assert out_vert.shape[1] == 2

            self.out_vert = out_vert
            self.out_vert_num = self.out_vert.shape[0]
            vert = np.vstack([vert, out_vert])
        else:
            self.out_vert = None
            self.out_vert_num = 0

        super().__init__(vert, face)

    def update_out_vert(self, out_vert: np.ndarray):
        assert self.out_vert is None
        assert out_vert.ndim == 2
        assert out_vert.shape[1] == 2

        self.out_vert = out_vert
        self.out_vert_num = self.out_vert.shape[0]
        self.vert = np.vstack([self.vert, self.out_vert])
        self.vert_num = self.vert.shape[0]

        ori_face_num = self.face.shape[0]
        out_face = Delaunay(np.vstack([self.circle, out_vert])).simplices
        out_face[out_face >= self.circle_num] += self.in_vert_num

        self.face = np.vstack([self.face, out_face])
        self.face_center = self.get_face_center()

        out_face_num = out_face.shape[0]
        idx = (
            np.where(np.linalg.norm(self.face_center[-out_face_num:], axis=1) <= 1)[0]
            + ori_face_num
        )

        self.face = np.delete(self.face, idx, axis=0)
        self.face_num = self.face.shape[0]
        self.face_center = np.delete(self.face_center, idx, axis=0)

        self.edges = self.get_edge()
        self.area = self.get_area()
        self.set_operator()


def get_rect(
    height: float | int = 1, width: float | int = 1, density: float | int = 0.01
) -> Mesh:
    """
    Create a rectangle mesh
    :param height: height of the rectangle
    :param width: width of the rectangle
    :param density: density of the mesh
    :return: face and vert arrays
    """
    x_num = int(width / density) + 1
    y_num = int(height / density) + 1
    x, y = np.meshgrid(np.arange(x_num), np.arange(y_num))
    x = x * density
    y = y * density
    vert = np.vstack([x.ravel(), y.ravel()]).T
    mesh = Mesh(vert)
    return mesh


def get_unit_disk(
    density: float | int = 0.01, circle_point_num: int = 1000, eps: float | None = None
) -> DiskMesh:
    """
    Create a unit disk mesh
    :param density: density of the mesh
    :param circle_point_num: number of points on the circle
    :param eps: epsilon to shrink the disk
    :return: face and vert arrays
    """
    if eps is None:
        eps = 10 / circle_point_num

    num = int(1 / density)
    x, y = np.meshgrid(np.arange(-num, num + 1), np.arange(-num, num + 1))
    vert = np.vstack([x.ravel(), y.ravel()]).T * density
    index = np.linalg.norm(vert, axis=1) <= 1 - eps
    vert = vert[index, :]

    if circle_point_num:
        circle = np.array(
            [
                [np.cos(t), np.sin(t)]
                for t in np.linspace(0, 2 * np.pi, circle_point_num + 1)
            ]
        )[:-1]

    mesh = DiskMesh(circle, vert)
    return mesh


def get_unit_disk_in_rect(
    disk_mesh: DiskMesh,
    height: float | int = 4,
    width: float | int = 4,
    density: float | int = 0.01,
    eps: float = 1e-3,
) -> DiskMesh:
    """
    Create a rectangle mesh from a unit disk mesh
    :param disk_mesh: unit disk mesh
    :param height: height of the rectangle
    :param width: width of the rectangle
    :param density: density of the mesh
    :return: combined mesh
    """
    x_num = int(width / density) + 1
    y_num = int(height / density) + 1
    x, y = np.meshgrid(np.arange(x_num), np.arange(y_num))
    x = x * density - width / 2
    y = y * density - height / 2
    out_vert = np.vstack([x.ravel(), y.ravel()]).T
    out_vert = out_vert[np.linalg.norm(out_vert, axis=1) > 1 + eps, :]

    new_disk_mesh = DiskMesh(disk_mesh.circle, disk_mesh.in_vert)
    new_disk_mesh.update_out_vert(out_vert)
    return new_disk_mesh
