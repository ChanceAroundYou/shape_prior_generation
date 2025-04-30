from contextlib import contextmanager

import numpy as np
from matplotlib import pyplot as plt
import cv2

plt.style.use("default")


@contextmanager
def figure_context(figsize=None, title=None, dpi=100):
    """Context manager for creating and cleaning up matplotlib figures"""
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    fig.tight_layout()
    if title:
        # ax.set_title(title, fontsize=16)
        fig.suptitle(title, fontsize=16)
    yield fig, ax
    plt.close(fig)


def fig_to_image(fig):
    fig.canvas.draw()
    image = np.array(fig.canvas.renderer.buffer_rgba())
    return image

def contour_to_binary(contour, image_size=256, rate=0.75):
    image = np.ones((image_size, image_size), dtype=np.uint8) * 255
    contour = contour.astype(np.float32)

    # Scale to fit within output size if needed
    min_pos = contour.min(axis=0)
    max_pos = contour.max(axis=0)
    # Take the larger dimension to maintain aspect ratio
    size = (max_pos - min_pos + 1e-5).max()
    # Scale to rate of image_size and center
    scaled_contour = (contour - min_pos) / size * (image_size * rate)
    scaled_contour += image_size / 2 - scaled_contour.mean(axis=0)

    scaled_contour = scaled_contour.astype(np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(image, [scaled_contour], color=0)
    image = np.repeat(image[:, :, np.newaxis], 3, axis=2)  # Convert to 3 channels
    return image


def contour_to_scatter(contour, color="b", pointsize=0.3, figsize=None, title=None, dpi=100):
    with figure_context(figsize=figsize, title=title, dpi=dpi) as (fig, ax):
        ax.scatter(contour[:, 0], contour[:, 1], s=pointsize, color=color)
        image = fig_to_image(fig)
        return image

def make_grid(images, rows, cols=None, figsize=None, title=None, dpi=100):
    n = len(images)
    h, w, c = images[0].shape
    # for image in images:
    #     if image.shape != (h, w, c):
    #         raise ValueError("All images must have the same shape")

    if cols is None:
        cols = int(np.ceil(n / rows))
        
    if figsize is None:
        title_height = 40 / dpi if title else 0
        figsize = (w * cols / dpi, h * rows / dpi + title_height)

    # Create background array for empty slots
    bg_value = 255
    empty_image = np.full((h, w, c), bg_value, dtype=np.uint8)
    if c == 4:  # For RGBA, set alpha to 255
        empty_image[..., 3] = 255

    # Pad the images list with background arrays if needed
    padded_images = images.copy()
    for _ in range(rows * cols - len(padded_images)):
        padded_images.append(empty_image.copy())
    # Convert to numpy array and reshape to (rows, cols, h, w, c)
    image_grid = np.array(padded_images).reshape(rows, cols, h, w, c)
    # Concatenate images horizontally for each row
    rows_list = [np.concatenate(row, axis=1) for row in image_grid]
    grid = np.concatenate(rows_list, axis=0)

    # Add title if provided
    with figure_context(figsize=figsize, title=title, dpi=dpi) as (fig, ax):
        ax.imshow(grid)
        grid = fig_to_image(fig)

    return grid
