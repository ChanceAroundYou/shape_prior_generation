import numpy as np

from hbs.mesh import Mesh
from ..utils.tool_functions import mu_chop, to_complex

def get_beltrami_coefficient(
    mapping: np.ndarray[np.floating], mesh: Mesh
) -> np.ndarray[np.complexfloating]:
    """
    Calculate the Beltrami coefficient of the mapping from "vertex" to "mapping", i.e., f(vertex) = map.
    :param mapping: n x 2 mapped vertex coordinates
    :return: Corresponding Beltrami coefficient
    """
    assert isinstance(mesh, Mesh), "mesh must be Mesh object"

    assert isinstance(mapping, np.ndarray) and np.issubdtype(
        mapping.dtype, np.floating
    ), "mapping must be float array"
    assert mapping.ndim == 2 and mapping.shape == (mesh.vert_num, 2), (
        "mapping must be n x 2 array and n is the number of faces"
    )

    mapping = to_complex(mapping)
    mu = (mesh.Dc * mapping) / (mesh.Dz * mapping)
    mu = mu_chop(mu)
    return mu
