import pytorch_lightning as pl
import torch
from torch import nn

from .components import *


def kl_loss(p, q, z):
    log_qz = q.log_prob(z)
    log_pz = p.log_prob(z)
    kl = log_qz - log_pz
    kl = kl.mean()
    return kl


class AutoEncoder(pl.LightningModule):
    def __init__(
            self,
            input_height: int,
            input_channel: int,
            enc_type: str = 'resnet18',
            first_conv: bool = False,
            maxpool1: bool = False,
            latent_dim: int = 256,
            lr: float = 1e-4,
            adam_beta1: float = 0.9,
            variational: bool = False,
            *args: any, **kwargs: any) -> None:
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()

        self.criterion_recon = torch.nn.MSELoss()
        self.criterion_kld = kl_loss

        valid_encoders = {
            'resnet9': {
                'enc': resnet9_encoder,
                'dec': resnet9_decoder,
            },
            'resnet18': {
                'enc': resnet18_encoder,
                'dec': resnet18_decoder,
            },
            'resnet50': {
                'enc': resnet50_encoder,
                'dec': resnet50_decoder,
            },
        }

        if enc_type not in valid_encoders:
            self.encoder = resnet18_encoder(
                input_channel, first_conv, maxpool1)
            self.decoder = resnet18_decoder(
                latent_dim, input_height, input_channel, first_conv, maxpool1)
        else:
            self.encoder = valid_encoders[enc_type]['enc'](
                input_channel, first_conv, maxpool1)
            self.decoder = valid_encoders[enc_type]['dec'](
                latent_dim, input_height, input_channel, first_conv, maxpool1)

        self.fc = nn.Linear(self.encoder.out_features, latent_dim *
                            2 if variational else latent_dim)

    def sample(self, x):
        latent_dim = self.hparams.latent_dim
        x = self.encoder(x)
        x = self.fc(x)
        mu = x[..., :latent_dim]
        lv = x[..., latent_dim:]

        std = torch.exp(lv / 2)
        p = torch.distributions.Normal(
            torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()

        return p, q, z

    def encode(self, x):
        if self.hparams.variational:
            p, q, z = self.sample(x)
        else:
            x = self.encoder(x)
            z = self.fc(x)
        return z

    def decode(self, z):
        return self.decoder.forward(z)

    def forward(self, z):
        return self.decode(z)

    def configure_optimizers(self):
        lr = self.hparams.lr
        beta1 = self.hparams.adam_beta1
        betas = (beta1, 0.999)

        return torch.optim.Adam(self.parameters(), lr=lr, betas=betas)

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        x, y = batch

        if self.hparams.variational:
            p, q, z = self.sample(x)
            x_hat = self.decode(z)

            loss_recon = self.criterion_recon(x_hat, x)
            loss_kld = self.criterion_kld(p, q, z) * 0.1
            loss = loss_recon + loss_kld

            # Logging
            self.log(f"{self.__class__.__name__}/recon", loss_recon)
            self.log(f"{self.__class__.__name__}/kld", loss_kld)
        else:
            z = self.encode(x)
            x_hat = self.decode(z)

            loss = self.criterion_recon(x_hat, x)

            self.log(f"{self.__class__.__name__}/recon", loss)

        return loss
