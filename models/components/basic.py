import numpy as np
import torch
from torch import Tensor, nn


class BasicEncoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        img_dim: tuple[int, int, int],
        first_hidden_dim: int = 256,
        normalize: bool = True,
        variational: bool = False,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.variational = variational

        def block(in_feat, out_feat, normalize: bool = normalize):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat))
            layers.append(nn.LeakyReLU(0.2))
            return layers

        self.fc = nn.Sequential(
            nn.Flatten(),
            *block(np.prod(img_dim), first_hidden_dim, False),
            *block(first_hidden_dim, first_hidden_dim // 2),
            *block(first_hidden_dim // 2, first_hidden_dim // 4),
            nn.Linear(first_hidden_dim // 4, latent_dim *
                      2 if variational else latent_dim),
        )

    def forward(self, x: Tensor):
        x = self.fc(x)

        if not self.variational:
            return x

        mu = x[..., :self.latent_dim]
        logvar = x[..., self.latent_dim:]
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        return mu + eps * std


class BasicDecoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        img_dim: tuple[int, int, int],
        first_hidden_dim: int = 256,
        normalize: bool = True,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        def block(in_feat, out_feat, normalize: bool = normalize):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat))
            layers.append(nn.LeakyReLU(0.2))
            return layers

        self.fc = nn.Sequential(
            *block(latent_dim, first_hidden_dim // 4),
            *block(first_hidden_dim // 4, first_hidden_dim // 2),
            *block(first_hidden_dim // 2, first_hidden_dim),
            nn.Linear(first_hidden_dim, np.prod(img_dim)),
            nn.Tanh(),
            nn.Unflatten(1, img_dim)
        )

    def forward(self, x: Tensor):
        x_hat = self.fc(x)
        return x_hat
