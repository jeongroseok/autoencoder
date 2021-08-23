import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data
import torchvision
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback
from torch import Tensor
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from models.autoencoder import AutoEncoder


class LatentSpaceVisualizer(Callback):
    def __init__(
        self,
        samples: int = 1000,
    ):
        super().__init__()
        self.samples = samples

    def on_epoch_end(self, trainer: Trainer, pl_module: AutoEncoder) -> None:
        writer: SummaryWriter = trainer.logger.experiment
        dataloader = trainer.train_dataloader

        features = torch.Tensor().to(pl_module.device)
        labels = torch.Tensor().to(pl_module.device)

        with torch.no_grad():
            pl_module.eval()
            for x, y in dataloader:
                x = x.to(pl_module.device)
                y = y.to(pl_module.device)
                remainings = max(0, self.samples - features.size(0))
                if remainings < 1:
                    break

                labels = torch.cat([labels, y[:remainings]], 0)
                features = torch.cat(
                    [features, pl_module.encoder(x)[:remainings]], 0)
                    
        pl_module.train()
        str_title = f'{pl_module.__class__.__name__}_latent_space'
        writer.add_embedding(features, metadata=labels,
                             tag=str_title, global_step=trainer.global_step)


class LatentDimInterpolator(Callback):
    def __init__(
        self,
        dataset,
        steps: int = 11,
    ):
        super().__init__()

        self.steps = steps
        self.dataset = dataset
        self.prepare_points()

    def prepare_points(self):
        x_0 = self.dataset[np.random.choice(len(self.dataset))][0]
        x_1 = self.dataset[np.random.choice(len(self.dataset))][0]
        self.points = torch.stack([x_0, x_1])

    def on_epoch_end(self, trainer: Trainer, pl_module: AutoEncoder) -> None:
        writer: SummaryWriter = trainer.logger.experiment
        num_rows = self.steps
        images = list(self.interpolate_latent_space(pl_module))
        images = torch.cat(images, 0)
        grid = torchvision.utils.make_grid(
            images, num_rows, normalize=True)

        str_title = f'{pl_module.__class__.__name__}_interpolation'
        writer.add_image(
            str_title, grid, global_step=trainer.global_step)

    def interpolate_latent_space(self, pl_module: AutoEncoder) -> list[Tensor]:
        points = self.points.to(pl_module.device)
        yield points[:1, ...]
        with torch.no_grad():
            pl_module.eval()
            z = pl_module.encoder(points)

            for w in np.linspace(0, 1, self.steps - 2):
                z_interpolated = torch.lerp(z[0], z[1], w).unsqueeze_(0)
                yield pl_module.decoder(z_interpolated)

        pl_module.train()
        yield points[1:, ...]
