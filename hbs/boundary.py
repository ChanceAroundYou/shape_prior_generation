import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline


def get_boundary(image_path, num_points=250, kernel_size=15):
    """
    Process the image to extract boundary, apply anti-aliasing, and resample.

    Parameters:
    image_path: Input image path
    num_points: Number of resampled points

    Returns:
    resampled_points: List of resampled boundary points
    """
    # Extract boundary points
    boundary_points = extract_boundary_points(image_path)
    # Apply anti-aliasing
    smoothed_points = anti_aliasing(boundary_points, kernel_size)
    # Resample to the specified number of points
    resampled_points = smooth_resample(smoothed_points, num_points)
    return resampled_points


def extract_boundary_points(image_path):
    """
    Extract ordered boundary points from a near black-and-white image.

    Parameters:
    image_path: Input image path

    Returns:
    boundary_points: List of ordered boundary points, format [(x1, y1), (x2, y2), ...]
    """
    # Read the image and convert to grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Add padding to avoid detecting image borders
    img = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=255)

    # Determine if the image is black-on-white or white-on-black
    # Calculate the mean color of the outer two layers of pixels
    outer_layer_mean = np.mean(
        [img[:2, :].mean(), img[-2:, :].mean(), img[:, :2].mean(), img[:, -2:].mean()]
    )

    if outer_layer_mean > 127:
        # White-on-black image, invert it
        img = cv2.bitwise_not(img)

    # Binarize the image
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # Find contours (using CHAIN_APPROX_NONE to get all boundary points)
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # Get the largest contour (assuming there's only one main shape in the image)
    main_contour = max(contours, key=cv2.contourArea)

    # Return the contour points directly, without sorting or anti-aliasing
    return main_contour.reshape(-1, 2)


def anti_aliasing(points, kernel_size=5):
    """
    Apply anti-aliasing to boundary points.

    Parameters:
    points: List of original boundary points

    Returns:
    smoothed_points: List of smoothed boundary points
    """
    # Use Gaussian blur for smoothing
    x = cv2.GaussianBlur(points[:, 0].astype("float64"), (kernel_size, kernel_size), 0)
    y = cv2.GaussianBlur(points[:, 1].astype("float64"), (kernel_size, kernel_size), 0)
    return np.column_stack((x, y))


def plot_boundary(boundary, with_line=True, scale=1):
    num_points = boundary.shape[0]
    x_min, x_max = np.min(boundary[:, 0]), np.max(boundary[:, 0])
    y_min, y_max = np.min(boundary[:, 1]), np.max(boundary[:, 1])
    aspect_ratio = (x_max - x_min) / (y_max - y_min)

    fig_size = (num_points / 60 * scale * aspect_ratio, num_points / 60 * scale)
    plt.figure(figsize=fig_size)
    if with_line:
        plt.plot(
            boundary[:, 0],
            boundary[:, 1],
            linestyle="-",
            linewidth=1 * scale,
            color="white",
            label="Boundary Line",
        )
    plt.scatter(
        boundary[:, 0],
        boundary[:, 1],
        marker="o",
        s=15 * scale,
        edgecolor="lightgreen",
        facecolor="none",
        label="Boundary Points",
    )
    plt.show()


def smooth_resample(points, num_points=None):
    """
    Smoothly and uniformly resample the boundary points matrix, generating a matrix of m points.
    :param points: nx2 matrix, original boundary point coordinates
    :param num_points: Number of resampled points
    :return: mx2 matrix, resampled boundary points
    """
    if num_points is None:
        num_points = points.shape[0]

    # points = np.unique(points, axis=0)
    _, idx = np.unique(points, axis=0, return_index=True)
    points = points[np.sort(idx)]

    # Check if the boundary is closed
    is_closed = np.all(np.abs(points[0, :] - points[-1, :]) < 1e-10)
    if not is_closed:
        points = np.vstack([points, points[0, :]])  # Close the boundary if not closed

    # Calculate cumulative arc length
    diffs = np.diff(points, axis=0)
    dists = np.sqrt(np.sum(diffs**2, axis=1))
    cum_dists = np.concatenate([[0], np.cumsum(dists)])
    total_length = cum_dists[-1]

    # Spline interpolation settings
    t = cum_dists
    x = points[:, 0]
    y = points[:, 1]

    ppx = CubicSpline(t, x, bc_type="periodic")
    ppy = CubicSpline(t, y, bc_type="periodic")
    t_new = np.linspace(0, total_length, num_points + 1)

    # Calculate new points
    x_new = ppx(t_new)
    y_new = ppy(t_new)
    resampled_points = np.column_stack([x_new, y_new])
    return resampled_points[:-1, :]


if __name__ == "__main__":
    # 示例用法
    boundary_points = extract_boundary_points("test_image.png")
    print(f"提取的边界点数量：{len(boundary_points)}")
