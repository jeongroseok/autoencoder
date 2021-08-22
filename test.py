import matplotlib.pyplot as plt
from trainer import AutoEncoder
from pl_examples import _DATASETS_PATH
import pl_bolts.datamodules.cifar10_datamodule
import torch

def main():
    dm = pl_bolts.datamodules.cifar10_datamodule.CIFAR10DataModule(_DATASETS_PATH, num_workers=1, batch_size=4)
    dm.setup()
    dl = dm.train_dataloader()
    batches = list(iter(dl))
    batch = batches[0]
    images = batch[0]
    labels = batch[1]

    image: torch.Tensor = images[0]
    label: torch.Tensor = labels[0]

    plt.imshow(image.permute(1,2,0)) # c, h, w to h, w, c
    plt.show()

    model = AutoEncoder().load_from_checkpoint(
        'lightning_logs\\version_3\\checkpoints\\epoch=6-step=8749.ckpt')
    encoder = model.encoder
    fc = model.fc
    decoder = model.decoder

    z: torch.Tensor = fc(encoder(image.unsqueeze(0)))
    z_new = z.clone()

    output = decoder(z)
    plt.imshow(output.squeeze(0).permute((1,2,0)).detach().numpy())
    plt.show()
    print('done')

if __name__ == "__main__":
    main()
