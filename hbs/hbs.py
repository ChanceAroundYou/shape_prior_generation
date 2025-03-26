import numpy as np

from hbs.conformal_welding import ConformalWelding, get_conformal_welding
from hbs.mesh import DiskMesh, get_unit_disk

from .qc import get_beltrami_coefficient, lsqc_solver
from .utils.geodesic_welding import geodesic_welding
from .utils.poisson import integral as poisson_integral
from .utils.tool_functions import to_complex, to_real


def get_hbs(
    bound: np.ndarray[np.floating],
    circle_point_num: int = 500,
    density: float = 0.01,
    disk: DiskMesh = None,
) -> tuple[
    np.ndarray[np.complexfloating],
    np.ndarray[np.floating],
    ConformalWelding,
    DiskMesh,
]:
    """
    Get Beltrami coefficients from boundary points
    :param `bound`: `n x 2` array, boundary points
    :param `circle_point_num`: number of points on the circle
    :param `density`: density of the mesh
    :param `disk`: DiskMesh object
    :return:
        `hbs`: `n` x 1 complex array, Beltrami coefficients defined on triangles
        `he`: `n` x 1 float array, harmonice extension of `hbs`
        `cw`: ConformalWelding object
        `disk`: DiskMesh object
    """
    if disk is None:
        disk = get_unit_disk(density, circle_point_num)

    cw = get_conformal_welding(bound)

    r = 0
    while True:
        cw.rotate_x(r / 2)
        cw.linear_interp(circle_point_num)
        he = poisson_integral(disk.in_vert, cw.x, cw.y)
        hbs = get_beltrami_coefficient(he, disk)

        excluded_idx = np.isnan(hbs) + np.linalg.norm(disk.face_center, axis=1) == 0
        r = np.angle(np.sum(hbs[~excluded_idx]))

        if abs(r) <= 5e-3:
            he_angle = np.angle(np.sum(hbs * to_complex(disk.face_center)))
            if he_angle < 0 or he_angle == np.pi:
                r += 2 * np.pi
            else:
                break

    return hbs, he, cw, disk


def reconstruct_from_hbs(hbs: np.ndarray[np.complexfloating], disk: DiskMesh):
    """
    Reconstruct original shape from HBS
    :param `hbs`: complex array with length `disk.face_num`, Beltrami coefficients defined on triangles
    :param `disk`: DiskMesh object
    :return:
        `bound_points`: `disk.circle_num` x 2 array, boundary points
        `in_points`: `disk.in_vert_num` x 2 array, inner points
        `out_points`: `disk.out_vert_num` x 2 array, outer points
        `mapping`: `disk.vert_num` x 2 array, vertex coordinates of the solved mapping
    """
    assert isinstance(disk, DiskMesh), "`disk` must be DiskMesh object"

    assert isinstance(hbs, np.ndarray) and np.issubdtype(
        hbs.dtype, np.complexfloating
    ), "`hbs` must be complex array"
    assert hbs.ndim == 1 and hbs.shape[0] == disk.face_num, (
        "the length of `hbs` is `n` and `n` is the number of faces"
    )

    target = np.array([[1, 0], [0, 0]], dtype=np.float64)

    one_pos = np.all(disk.vert == target[0], axis=1)
    zero_pos = np.all(disk.vert == target[1], axis=1)
    landmark = np.where(one_pos | zero_pos)[0]

    mapping = lsqc_solver(hbs, landmark, target, disk)
    bound_points = mapping[: disk.circle_num]
    in_points = mapping[: disk.in_vert_num + disk.circle_num]
    out_points = mapping[disk.in_vert_num + disk.circle_num :]

    in_points, out_points = geodesic_welding(
        to_complex(in_points),
        to_complex(out_points),
        to_complex(bound_points),
        to_complex(disk.circle),
    )
    in_points = to_real(in_points)
    out_points = to_real(out_points)

    bound_points = in_points[: disk.circle_num]
    in_points = in_points[disk.circle_num :]
    return bound_points, in_points, out_points, mapping
    return bound_points, in_points, out_points, mapping
