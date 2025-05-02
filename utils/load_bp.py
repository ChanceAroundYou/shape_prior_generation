import os

import numpy as np

from hbs.boundary import get_boundary


def load_from_img(img_path: str, bound_point_num=100, kernel_size=15) -> np.ndarray:
    bound = get_boundary(img_path, bound_point_num, kernel_size)
    
    # Normalize boundary points to range [0, 1]
    if bound is not None and len(bound) > 0:
        min_vals = np.min(bound, axis=0)
        max_vals = np.max(bound, axis=0)
        size = (max_vals - min_vals).max() + 1e-10
        bound = (bound - min_vals) / size  # Adding small epsilon to avoid division by zero
    return bound


def load_from_dir(
    img_dir: str,
    bound_point_num=100,
    kernel_size=15,
    exclude_list=[],
) -> np.ndarray:
    bound_dict = {}
    for img_name in os.listdir(img_dir):
        if not img_name.endswith(".png") and not img_name.endswith(".jpg"):
            continue

        if img_name in exclude_list:
            continue

        print(f"Loading {img_name}")
        img_path = os.path.join(img_dir, img_name)
        bound_dict[img_name] = load_from_img(img_path, bound_point_num, kernel_size)

    result = np.stack([bd for _, bd in bound_dict.items()])
    # result = result.flatten()
    return result
