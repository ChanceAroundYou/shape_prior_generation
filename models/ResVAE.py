import torch
from torch import nn

from models.base_VAE import BaseVAE


class ResLinearBlock(nn.Module):
    def __init__(self, in_features, out_features, hidden_features, repeat=0, bias=True):
        super(ResLinearBlock, self).__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(in_features, hidden_features, bias=bias),
            nn.ReLU(),
            # nn.BatchNorm1d(hidden_features)
            nn.LayerNorm(hidden_features),
        )
        self.hidden_layers = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Linear(hidden_features, hidden_features, bias=bias),
                    nn.ReLU(),
                    # nn.BatchNorm1d(hidden_features)
                    nn.LayerNorm(hidden_features),
                )
                for _ in range(repeat)
            ]
        )
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_features, out_features, bias=bias),
            nn.ReLU(),
            # nn.BatchNorm1d(out_features)
            nn.LayerNorm(out_features),
        )
        if in_features != out_features:
            self.shortcut = nn.Linear(in_features, out_features, bias=bias)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        x1 = self.input_layer(x)
        x1 = self.hidden_layers(x1)
        x1 = self.output_layer(x1)

        x = self.shortcut(x)
        x = torch.relu(x + x1)
        return x


class ResVAEEncoder(nn.Module):
    def __init__(self, input_dim, output_dim=2, hidden_dim=200, hidden_layers=[2, 2]):
        super().__init__()
        self.encoder_layers = nn.Sequential(
            *[
                (
                    ResLinearBlock(hidden_dim, hidden_dim, hidden_dim, repeat=layer_num)
                    if i > 0
                    else ResLinearBlock(
                        input_dim, hidden_dim, hidden_dim, repeat=layer_num
                    )
                )
                for i, layer_num in enumerate(hidden_layers)
            ]
        )
        self.fc_mu = nn.Linear(hidden_dim, output_dim)
        self.fc_logvar = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.encoder_layers(x)
        mu = self.fc_mu(x)
        log_var = self.fc_logvar(x)
        return mu, log_var


class ResVAEDecoder(nn.Module):
    def __init__(
        self, input_dim=2, output_dim=100, hidden_dim=200, hidden_layers=[2, 2]
    ):
        super().__init__()
        self.decoder_layers = nn.Sequential(
            *[
                (
                    ResLinearBlock(hidden_dim, hidden_dim, hidden_dim, repeat=layer_num)
                    if i > 0
                    else ResLinearBlock(
                        input_dim, hidden_dim, hidden_dim, repeat=layer_num
                    )
                )
                for i, layer_num in enumerate(hidden_layers)
            ]
        )
        self.fc_output = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.decoder_layers(x)
        x = self.fc_output(x)
        return x


class ResVAE(BaseVAE):
    def __init__(self, input_dim, hidden_dim=200, hidden_layers=[2, 2], latent_dim=2, device="cpu"):
        super().__init__(input_dim, latent_dim, device)
        self.encoder = ResVAEEncoder(input_dim, latent_dim, hidden_dim, hidden_layers)
        self.decoder = ResVAEDecoder(latent_dim, input_dim, hidden_dim, hidden_layers)
        self.to(device)


if __name__ == "__main__":
    from torchsummary import summary

    model = ResVAE(100, hidden_dim=1000, hidden_layers=[1], latent_dim=2)
    summary(model, (100,), device="cpu")
