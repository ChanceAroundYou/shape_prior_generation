from models.ResVAE import ResVAE
import torch

class BPResVAE(ResVAE):
    def reconstruct(self, h):
        h = h.view(-1, self.input_dim // 2, 2)
        return h
    
    # def forward(self, x):
    #     x = x.view(-1, self.input_dim)
    #     return super().forward(x)
    
    def loss(self, x, ground_truth, mu, log_var, kl_rate=0.5):
        # Reconstruction loss
        recon_loss = (self.reconstruct(ground_truth) - self.reconstruct(x)).pow(2).sum(dim=1).mean()

        # KL divergence loss
        kl_loss = mu.pow(2) + log_var.exp() - log_var - 1
        kl_loss = 0.5 * torch.sum(kl_loss, dim=1).mean()
        loss = recon_loss + kl_rate * kl_loss
        return {
            "loss": loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss
        }