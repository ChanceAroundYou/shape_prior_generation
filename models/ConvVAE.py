import torch
# import torch.nn.functional as F
from torch import nn
import math


class ResConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, sample=False, out_padding=0, transpose=False):
        super().__init__()
        self.sample = sample
        self.stride = 2 if sample else 1
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.transpose = transpose
        self.out_padding = out_padding
        
        self.layer = torch.nn.Sequential(
            self.get_conv(in_channels, hidden_channels, 5, self.stride, 2),
            torch.nn.BatchNorm1d(hidden_channels),
            torch.nn.ReLU(),
            nn.Conv1d(hidden_channels, hidden_channels, 5, 1, 2),
            torch.nn.BatchNorm1d(hidden_channels),
            torch.nn.ReLU(),
            nn.Conv1d(hidden_channels, out_channels, 5, 1, 2),
            torch.nn.BatchNorm1d(out_channels),
            torch.nn.ReLU(),
        )
        if self.in_channels != self.out_channels or self.sample:
            self.res_layer = self.get_conv(in_channels, out_channels, 1, self.stride)

    def get_conv(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0):
        if self.transpose:
            out_padding = self.out_padding if stride > 1 else 0
            return torch.nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding, output_padding=out_padding)
        else:
            return torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        if self.in_channels != self.out_channels or self.sample:
            res = self.res_layer(x)
        else:
            res = x
        return self.layer(x) + res

class Encoder(nn.Module):
    def __init__(self, in_dim=100, z_dim=2, h_channels=16, layers=[2,2]):
        super().__init__()
        layer_num = len(layers)
        self.layers = layers
        self.in_dim = in_dim
        self.z_dim = z_dim
        self.h_channels = h_channels
        self.feature_channels = h_channels * (2 ** layer_num)
        self.feature_dim = math.ceil(in_dim/(2 ** (layer_num+1)))
        
        self.input_layer = nn.Sequential(
            nn.Conv1d(1, h_channels, kernel_size=7, padding=3, stride=2),
            # nn.MaxPool1d(2)
        )

        self.encoder_layers =nn.Sequential(*[
            nn.Sequential(*[
                ResConvBlock(
                    h_channels * (2 ** (i+int(j>0))), 
                    h_channels * (2 ** (i+1)), 
                    h_channels * (2 ** (i+1)), 
                    j==0
                ) 
                for j in range(layer)
            ])
            for i, layer in enumerate(layers)
        ])

        self.mu_layer = nn.Linear(self.feature_channels * self.feature_dim, z_dim)
        self.log_var_layer = nn.Linear(self.feature_channels * self.feature_dim, z_dim)
        
    def forward(self, x):
        x = self.input_layer(x)
        x = self.encoder_layers(x)
        x = x.flatten(1)
        mu = self.mu_layer(x)
        log_var = self.log_var_layer(x)
        return mu, log_var

class Decoder(nn.Module):
    def __init__(self, out_dim=100, z_dim=2, h_channels=16, layers=[2,2]):
        super().__init__()
        layer_num = len(layers)
        self.layers = layers
        self.out_dim = out_dim
        self.z_dim = z_dim
        self.h_channels = h_channels
        self.feature_channels = h_channels * (2 ** layer_num)
        self.feature_dim = math.ceil(out_dim/(2 ** (layer_num+1)))

        self.input_layer = nn.Sequential(
            nn.Linear(z_dim, self.feature_channels * self.feature_dim),
            nn.ReLU()
        )
        self.decoder_layers = nn.Sequential(*[
            nn.Sequential(*[
                ResConvBlock(
                    h_channels * (2 ** (layer_num-i)), 
                    h_channels * (2 ** (layer_num-i-int(j==(layer-1)))), 
                    h_channels * (2 ** (layer_num-i)),
                    j==(layer-1),
                    1 if j==(layer-1) and math.ceil(out_dim / (2 ** (layer_num-i))) % 2 == 0 else 0,
                    True
                )
                for j in range(layer)
            ])
            for i, layer in enumerate(layers)
        ])
        self.output_layer = nn.Sequential(
            nn.ConvTranspose1d(h_channels, 1, kernel_size=7, padding=3, stride=2, output_padding=1-out_dim%2),
        )
        
        
    def forward(self, z):
        x = self.input_layer(z)
        x = x.view(-1, self.feature_channels, self.feature_dim)
        # print(x.shape)
        # print(self.decoder_layers)
        x = self.decoder_layers(x)
        x = self.output_layer(x)
        x = x.view(-1, self.out_dim)
        return x

class ResConvVAE(nn.Module):
    def __init__(self, cw_dim=100, z_dim=2, h_dim=16, h_layers=[1,1]):
        super().__init__()
        # encoder
        self.cw_dim = cw_dim
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.h_layers = h_layers
        
        self.encoder = Encoder(cw_dim, z_dim, h_dim, h_layers)

        # decoder
        self.decoder = Decoder(cw_dim, z_dim, h_dim, h_layers)

        # Define the learnable alpha parameter
        self.mean = nn.Parameter(torch.tensor(3.168751), requires_grad=True)
        self.std = nn.Parameter(torch.tensor(1.014383), requires_grad=True)

    # def encode(self, x):
        
    #     mu, log_var = self.encoder(x)
    #     return mu, log_var

    # def decode(self, z, eps=1e-8):
    #     x = self.decoder(x)
    #     return x

    def forward(self, x):
        x = x.view(-1, 1, self.cw_dim)
        mu, log_var = self.encoder(x)
        x_reconstructed = self.generate(mu, log_var)
        return x_reconstructed, mu, log_var
    
    def generate(self, mu, log_var):
        sigma = log_var.exp().sqrt()
        epsilon = torch.randn_like(sigma)
        z = mu + sigma * epsilon
        x = self.decoder(z)
        x = (x - x.mean()) / x.std()
        x = x * self.std + self.mean
        return x
    
    def reconstruct(self, h, eps=1e-10):
        h = 1 / (torch.exp(h)+eps)
        h = h / torch.sum(h, dim=1, keepdim=True) * 2 * torch.pi
        return h
    
    def loss(self, x, x_reconstructed, mu, log_var, kl_rate=0.01):
        # Reconstruction loss
        # recon_loss = F.binary_cross_entropy(x_reconstructed, x, reduction='sum')
        recon_loss = (x_reconstructed - x).pow(2).sum(dim=1).mean()
        true_loss = (self.reconstruct(x) - self.reconstruct(x_reconstructed)).pow(2).sum(dim=1).mean()

        # KL divergence loss
        kl_loss = mu.pow(2) + log_var.exp() - log_var - 1
        kl_loss = 0.5 * torch.sum(kl_loss, dim=1).mean()
        loss = recon_loss + kl_rate * kl_loss
        return {
            'loss': loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'true_loss': true_loss
        }
    
    def initial(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d, nn.ConvTranspose1d)):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                
if __name__ == '__main__':
    from torchsummary import summary
    
    model = ResConvVAE(100, 2, 32, [2,2])
    
    summary(model, (1, 100), device='cpu')
    # print(mu.shape, log_var.shape)