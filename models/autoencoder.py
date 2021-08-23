import pytorch_lightning as pl
import torch
from torch.nn import functional as F
from torch import nn

from .components.resnet import *


class AutoEncoder(pl.LightningModule):
    def __init__(
            self,
            img_dim: tuple[int, int, int],
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

        self.criterion_recon = nn.MSELoss()
        self.criterion_kld = nn.KLDivLoss(reduction='batchmean')

        self.encoder = create_encoder(
            enc_type, latent_dim, img_dim, first_conv, maxpool1, variational)
        self.decoder = create_decoder(
            enc_type, latent_dim, img_dim, first_conv, maxpool1)

    def forward(self, z):
        return self.decoder(z)

    def configure_optimizers(self):
        lr = self.hparams.lr
        beta1 = self.hparams.adam_beta1
        betas = (beta1, 0.999)

        return torch.optim.Adam(self.parameters(), lr=lr, betas=betas)

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        x, y = batch

        z = self.encoder(x)
        x_hat = self.decoder(z)

        if self.hparams.variational:

            loss_recon = self.criterion_recon(x_hat, x)
            loss_kld = self.criterion_kld(
                F.log_softmax(z, 1), torch.ones_like(z)) * 0.1
            loss = loss_recon + loss_kld

            self.log(f"{self.__class__.__name__}/recon", loss_recon)
            self.log(f"{self.__class__.__name__}/kld", loss_kld)
        else:
            loss = self.criterion_recon(x_hat, x)

            self.log(f"{self.__class__.__name__}/recon", loss)

        return loss
