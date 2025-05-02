from contextlib import contextmanager

import numpy as np
from matplotlib import pyplot as plt
import cv2

plt.style.use("default")


@contextmanager
def figure_context(figsize=None, title=None, dpi=100, image_size=256):
    """Context manager for creating and cleaning up matplotlib figures"""
    if figsize is None:
        title_height = 40 / dpi if title else 0
        figsize = (image_size / dpi, image_size / dpi + title_height)
    
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    fig.tight_layout()
    if title:
        fig.suptitle(title, fontsize=16)
    yield fig, ax
    plt.close(fig)


def fig_to_image(fig):
    fig.canvas.draw()
    image = np.array(fig.canvas.renderer.buffer_rgba())
    return image

def contour_to_binary(contour, rate=0.75, figsize=None, title=None, dpi=100, color=(0, 0, 0)):
    image_size = 256
    image = np.ones((image_size, image_size), dtype=np.uint8) * 255
    contour = contour.astype(np.float32)

    min_pos = contour.min(axis=0)
    max_pos = contour.max(axis=0)
    size = (max_pos - min_pos + 1e-5).max()
    scaled_contour = (contour - min_pos) / size * (image_size * rate)
    scaled_contour += image_size / 2 - scaled_contour.mean(axis=0)

    scaled_contour = scaled_contour.astype(np.int32).reshape((-1, 1, 2))
    image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
    cv2.fillPoly(image, [scaled_contour], color=color)
    
    with figure_context(figsize=figsize, title=title, dpi=dpi, image_size=image_size) as (fig, ax):
        ax.imshow(image)
        image = fig_to_image(fig)
    
    return image


def contour_to_scatter(contour, color="b", pointsize=0.3, figsize=None, title=None, dpi=100):
    with figure_context(figsize=figsize, title=title, dpi=dpi) as (fig, ax):
        ax.scatter(contour[:, 0], contour[:, 1], s=pointsize, color=color)
        image = fig_to_image(fig)
        return image

def make_grid(images, rows, cols=None, figsize=None, title=None, dpi=100):
    n = len(images)
    h, w, c = images[0].shape

    if cols is None:
        cols = int(np.ceil(n / rows))

    bg_value = 255
    empty_image = np.full((h, w, c), bg_value, dtype=np.uint8)
    if c == 4:
        empty_image[..., 3] = 255

    padded_images = images.copy()
    for _ in range(rows * cols - len(padded_images)):
        padded_images.append(empty_image.copy())
    image_grid = np.array(padded_images).reshape(rows, cols, h, w, c)
    rows_list = [np.concatenate(row, axis=1) for row in image_grid]
    grid = np.concatenate(rows_list, axis=0)

    with figure_context(figsize=figsize, title=title, dpi=dpi, image_size=h) as (fig, ax):
        ax.imshow(grid)
        grid = fig_to_image(fig)

    return grid

def visualize_latent_space(z_ground_truth, z_generated, annotate=True):
    z_generated = z_generated.cpu().detach().numpy()
    z_ground_truth = z_ground_truth.cpu().detach().numpy()
    plt.scatter(z_generated[:, 0], z_generated[:, 1], color='b')
    if annotate:
        for i, (x, y) in enumerate(z_ground_truth):
            plt.annotate(str(i), (x, y), xytext=(2, 2), textcoords='offset points')
    plt.scatter(z_ground_truth[:, 0], z_ground_truth[:, 1], color="r")
    plt.title("Latent Space Distribution\n(Red: Original, Blue: Generated)")
    plt.show()

def visualize_cw_comparison(generated_cw, ground_truth_cw, figsize=(10, 5)):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(122)
    ax.title.set_text("Ground Truth Conformal Welding")
    for i in range(len(ground_truth_cw)):
        ax.plot(ground_truth_cw[i])
        
    ax = fig.add_subplot(121)
    ax.title.set_text("Generated Conformal Welding")
    for i in range(len(generated_cw)):
        ax.plot(generated_cw[i])

    