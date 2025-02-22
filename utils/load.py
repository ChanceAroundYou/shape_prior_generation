from typing import List

import os
import h5py
from hbs.conformal_welding import ConformalWelding
import numpy as np
from hbs import get_hbs
from hbs.boundary import get_boundary

# from hbs.conformal_welding import ConformalWelding


def load_cw(mat_path: str, class_names: List[str] = []) -> np.ndarray:
    conformal_weldings = []

    with h5py.File(mat_path, "r") as mat_file:
        for class_name in mat_file:
            if class_names and class_name not in class_names:
                continue

            class_group = mat_file[class_name]
            for case_name in class_group:
                theta = class_group[case_name]
                preprocessed_theta = loading_preprocess(theta)
                conformal_weldings.append(preprocessed_theta)

    conformal_weldings = np.array(conformal_weldings)
    return conformal_weldings


def load_from_img(img_path: str, bound_point_num=500, cw_point_num=100):
    bound = get_boundary(img_path, bound_point_num)
    hbs, he, cw, disk = get_hbs(bound, 1000, 0.01)
    cw.linear_interp(cw_point_num)
    return cw, bound

def load_from_dir(img_dir: str, bound_point_num=500, cw_point_num=100, exclude_list=[]):
    data = dict()
    for img_name in os.listdir(img_dir):
        if not img_name.endswith('.png') and not img_name.endswith('.jpg'):
            continue

        if img_name in exclude_list:
            continue

        print(img_name)
        img_path = os.path.join(img_dir, img_name)
        data[img_name] = load_from_img(img_path, bound_point_num, cw_point_num)
    
    result = np.stack([preprocess(cw) for _, (cw, _) in data.items()])
    return result, data

def preprocess(cw: ConformalWelding) -> np.ndarray:
    theta = cw.get_y_angle_diff()
    theta = np.log(1 / theta)
    return theta

def loading_preprocess(theta: np.ndarray) -> np.ndarray:
    theta = np.insert(theta, 100, 2 * np.pi)
    theta = np.diff(theta)
    theta = np.log(1 / theta)
    return theta
