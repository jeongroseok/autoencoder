import pytorch_lightning as pl
import torch

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
            latent_dim: int = 32,
            img_dim: tuple[int, int, int] = (1, 28, 28),
            lr: float = 1e-4,
            adam_beta1: float = 0.9,
            hidden_dim: int = 256,
            normalize: bool = True,
            variational: bool = False,
            *args: any, **kwargs: any) -> None:
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()

        self.criterion_recon = torch.nn.MSELoss()
        self.criterion_kld = kl_loss

        self.encoder = Encoder(latent_dim, img_dim,
                               hidden_dim, normalize, variational)
        self.decoder = Decoder(latent_dim, img_dim, hidden_dim, normalize)

    def encode(self, x):
        if self.hparams.variational:
            p, q, z = self.encoder.forward(x)
        else:
            z = self.encoder.forward(x)
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
            p, q, z = self.encoder.forward(x)
            x_hat = self.decoder.forward(z)

            loss_recon = self.criterion_recon(x_hat, x)
            loss_kld = self.criterion_kld(p, q, z) * 0.1
            loss = loss_recon + loss_kld

            # Logging
            self.log(f"{self.__class__.__name__}/recon", loss_recon)
            self.log(f"{self.__class__.__name__}/kld", loss_kld)
        else:
            z = self.encoder.forward(x)
            x_hat = self.decoder.forward(z)

            loss = self.criterion_recon(x_hat, x)

            self.log(f"{self.__class__.__name__}/recon", loss)

        return loss
