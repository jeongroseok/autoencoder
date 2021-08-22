import pytorch_lightning as pl
from pl_bolts.datamodules import MNISTDataModule
from pl_examples import _DATASETS_PATH
from torchvision.datasets.mnist import MNIST

from callbacks import LatentDimInterpolator, LatentSpaceVisualizer
from models.autoencoder import AutoEncoder
from utils import set_persistent_workers


def main(args=None):
    set_persistent_workers(MNISTDataModule)
    datamodule = MNISTDataModule(_DATASETS_PATH, num_workers=4,
                                 batch_size=256, shuffle=False, drop_last=True)
    model = AutoEncoder(256, datamodule.dims, lr=1e-4,
                hidden_dim=2048, variational=False)  # VAE는 너무 작으면 학습이 안됨

    dataset = MNIST(_DATASETS_PATH, False,
                    transform=datamodule.default_transforms())
    callbacks = [
        LatentSpaceVisualizer(2500),
        LatentDimInterpolator(dataset),
    ]

    trainer = pl.Trainer(
        gpus=-1,
        progress_bar_refresh_rate=5,
        max_epochs=1000,
        callbacks=callbacks
    )
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
