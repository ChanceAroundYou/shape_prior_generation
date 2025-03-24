import numpy as np
import torch

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
    generated_data, z = generate_data(model, num, mu_vals, logvar_vals, k)
    generated_cw = reconstruct_cw(model, generated_data)
    return generated_cw, generated_data, z


def generate_shape(model, num=1, mu_vals=None, logvar_vals=None, k=1):
    generated_cw, generated_data, z = generate_cw(model, num, mu_vals, logvar_vals, k)
    generated_shape = reconstruct_shape(generated_cw)
    return generated_shape, generated_cw, generated_data, z


def reconstruct_cw(model, generated_data):
    generated_cw = model.reconstruct(generated_data)
    return generated_cw.numpy()


def reconstruct_shape(generated_cw):
    input_dim = generated_cw.shape[1]
    x_angle = np.linspace(0, 2 * np.pi, input_dim + 1)[:input_dim]
    x = np.exp(1j * x_angle)

    generated_shape = []
    for cw in generated_cw:
        y = np.exp(1j * cw)
        try:
            shape, _ = geodesicwelding(y, [], y, x)
        except Exception as e:
            print(e)
            shape = np.zeros_like(x)
        generated_shape.append(shape)
    # generated_shape = [geodesicwelding(np.exp(1j * y), [], np.exp(1j * y), x)[0] for y in generated_cw.numpy()]
    return generated_shape
