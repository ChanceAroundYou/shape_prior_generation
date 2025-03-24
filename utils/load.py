import os
import pickle
from typing import Dict, List, Tuple

import h5py
import numpy as np
from hbs import get_hbs
from hbs.boundary import get_boundary
from hbs.conformal_welding import ConformalWelding

# from hbs.conformal_welding import ConformalWelding


# def load_cw(mat_path: str, class_names: List[str] = []) -> np.ndarray:
#     conformal_weldings = []

#     with h5py.File(mat_path, "r") as mat_file:
#         for class_name in mat_file:
#             if class_names and class_name not in class_names:
#                 continue

#             class_group = mat_file[class_name]
#             for case_name in class_group:
#                 theta = class_group[case_name]
#                 preprocessed_theta = loading_preprocess(theta)
#                 conformal_weldings.append(preprocessed_theta)

#     conformal_weldings = np.array(conformal_weldings)
#     return conformal_weldings


def load_from_img(img_path: str, bound_point_num=500, cw_point_num=100, kernel_size=15) -> ConformalWelding:
    bound = get_boundary(img_path, bound_point_num, kernel_size)
    hbs, he, cw, disk = get_hbs(bound, 1000, 0.01)
    cw.linear_interp(cw_point_num)
    return cw


def load_from_dir(
    img_dir: str, bound_point_num=500, cw_point_num=100, kernel_size=15, exclude_list=[], read_from_file=True
) -> Tuple[np.ndarray, Dict[str, ConformalWelding]]:
    cw_dict_path = os.path.join(img_dir, "cw_dict.pkl")
    if read_from_file and os.path.exists(cw_dict_path):
        cw_dict: Dict[str, ConformalWelding] = pickle.load(open(cw_dict_path, "rb"))
        print(f"Read from saved cw_dict file {cw_dict_path}")

        for _, cw in cw_dict.items():
            if len(cw.y) != cw_point_num:
                cw.linear_interp(cw_point_num)
    else:
        cw_dict: Dict[str, ConformalWelding] = dict()
        for img_name in os.listdir(img_dir):
            if not img_name.endswith(".png") and not img_name.endswith(".jpg"):
                continue

            if img_name in exclude_list:
                continue

            print(f"Loading {img_name}")
            img_path = os.path.join(img_dir, img_name)
            cw_dict[img_name] = load_from_img(img_path, bound_point_num, cw_point_num, kernel_size)
        
        with open(cw_dict_path, "wb") as f:
            pickle.dump(cw_dict, f)
            print(f"Save cw_dict to file {cw_dict_path}")

    result = np.stack([preprocess(cw) for _, cw in cw_dict.items()])
    return result, cw_dict


def preprocess(cw: ConformalWelding, eps=1e-8) -> np.ndarray:
    theta = cw.get_y_angle_diff()
    print(theta.dtype)
    theta[theta < eps] = eps
    theta = theta / np.sum(theta) * 2 * np.pi
    theta = np.log(1 / theta)
    return theta


# def loading_preprocess(theta: np.ndarray) -> np.ndarray:
#     theta = np.insert(theta, 100, 2 * np.pi)
#     theta = np.diff(theta)
#     theta = np.log(1 / theta)
#     return theta
