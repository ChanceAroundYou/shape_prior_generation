import torch
from torch import nn


class BaseVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, device="cpu"):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.device = device

        self.encoder = nn.Module()
        self.decoder = nn.Module()

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def reparameterize(self, mu, log_var, k=1):
        std = (0.5 * log_var).exp()
        eps = k * torch.randn_like(std)
        z = mu + std * eps
        return z

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x = self.decode(z)
        return x, mu, log_var

    def reconstruct(self, h, eps=1e-8):
        h = 1 / (torch.exp(h) + eps)
        h = h / torch.sum(h, dim=1, keepdim=True) * 2 * torch.pi
        h[h < eps] = eps
        h = h / torch.sum(h, dim=1, keepdim=True) * 2 * torch.pi
        h = h.cumsum(dim=1)
        return h

    def loss(self, x, x_reconstructed, mu, log_var, kl_rate=0.5, true_rate=1):
        # Reconstruction loss
        # recon_loss = F.binary_cross_entropy(x_reconstructed, x, reduction='sum')
        recon_loss = (x_reconstructed - x).pow(2).sum(dim=1).mean()
        true_loss = (
            (self.reconstruct(x) - self.reconstruct(x_reconstructed))
            .pow(2)
            .sum(dim=1)
            .mean()
        )

        # KL divergence loss
        kl_loss = mu.pow(2) + log_var.exp() - log_var - 1
        kl_loss = 0.5 * torch.sum(kl_loss, dim=1).mean()
        loss = recon_loss + true_rate * true_loss + kl_rate * kl_loss
        return {
            "loss": loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
            "true_loss": true_loss,
        }

    def initial(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                try:
                    nn.init.constant_(m.bias, 0)
                except Exception as e:
                    print(e)
