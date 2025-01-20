from typing import List

import h5py
import numpy as np


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


def loading_preprocess(theta: np.ndarray) -> np.ndarray:
    theta = np.insert(theta, 100, 2 * np.pi)
    theta = np.diff(theta)
    theta = np.log(1 / theta)
    return theta
