import pytorch_lightning as pl
from pl_bolts.datamodules import MNISTDataModule
from pl_examples import _DATASETS_PATH
from torchvision import transforms as transform_lib
from torchvision.datasets.mnist import MNIST

from callbacks import LatentDimInterpolator, LatentSpaceVisualizer
from models.autoencoder import AutoEncoder
from utils import set_persistent_workers


def main(args=None):
    set_persistent_workers(MNISTDataModule)
    input_height = 32
    input_channel = 1
    transforms = transform_lib.Compose([
        transform_lib.Resize(input_height),
        transform_lib.ToTensor(),
        transform_lib.Normalize(mean=(0.5, ), std=(0.5, )),
    ])
    datamodule = MNISTDataModule(
        _DATASETS_PATH,
        num_workers=4,
        batch_size=256,
        shuffle=False,
        drop_last=True,
        train_transforms=transforms,
        val_transforms=transforms,
        test_transforms=transforms,
    )

    model = AutoEncoder(
        input_height=input_height,
        input_channel=input_channel,
        latent_dim=16,
        variational=True
    )

    dataset = MNIST(_DATASETS_PATH, False,
                    transform=transforms)
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
