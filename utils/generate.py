import numpy as np
import torch

from hbs import boundary
from utils.geodesicwelding import geodesicwelding


def generate_data(model, num=1, mu_vals=None, logvar_vals=None, k=1):
    if mu_vals is None:
        mu = torch.zeros(num, model.latent_dim)
    elif isinstance(mu_vals, (int, float)):
        mu = mu_vals * torch.ones(num, model.latent_dim)
    elif isinstance(mu_vals, np.ndarray) and mu_vals.shape == (num, model.latent_dim):
        mu = torch.tensor(mu_vals)
    elif isinstance(mu_vals, torch.Tensor) and mu_vals.shape == (num, model.latent_dim):
        mu = mu_vals
    else:
        raise ValueError("Invalid mu_vals")

    if logvar_vals is None:
        logvar = torch.zeros(num, model.latent_dim)
    elif isinstance(logvar_vals, (int, float)):
        logvar = logvar_vals * torch.ones(num, model.latent_dim)
    elif isinstance(logvar_vals, np.ndarray) and logvar_vals.shape == (
        num,
        model.latent_dim,
    ):
        logvar = torch.tensor(logvar_vals)
    elif isinstance(logvar_vals, torch.Tensor) and logvar_vals.shape == (
        num,
        model.latent_dim,
    ):
        logvar = logvar_vals
    else:
        raise ValueError("Invalid logvar_vals")

    model.eval()
    with torch.no_grad():
        mu = mu.to(model.device, dtype=torch.float64)
        logvar = logvar.to(model.device, dtype=torch.float64)
        z = model.reparameterize(mu, logvar, k)
        generated_data = model.decode(z)

    return generated_data.cpu().detach(), z


def generate_cw(model, num=1, mu_vals=None, logvar_vals=None, k=1):
    data, z = generate_data(model, num, mu_vals, logvar_vals, k)
    cw = reconstruct_cw(model, data)
    return cw, data, z


def generate_contour(model, num=1, mu_vals=None, logvar_vals=None, k=1):
    cw, data, z = generate_cw(model, num, mu_vals, logvar_vals, k)
    contours = reconstruct_contour(cw)
    return contours, cw, data, z


def reconstruct_cw(model, generated_data):
    generated_cw = model.reconstruct(generated_data)
    return generated_cw.numpy()


def reconstruct_contour(generated_cw):
    input_dim = generated_cw.shape[1]
    x_angle = np.linspace(0, 2 * np.pi, input_dim + 1)[:input_dim]
    x = np.exp(1j * x_angle)

    contours = []
    for cw in generated_cw:
        y = np.exp(1j * cw)
        try:
            contour, _ = geodesicwelding(y, [], y, x)
        except Exception as e:
            print(e)
            contour = np.zeros_like(x)

        contour = np.stack([contour.real, contour.imag], axis=1)
        try:
            contour = boundary.smooth_resample(contour)
        except:
            pass

        contours.append(contour)
    # generated_shape = [geodesicwelding(np.exp(1j * y), [], np.exp(1j * y), x)[0] for y in generated_cw.numpy()]
    return contours
