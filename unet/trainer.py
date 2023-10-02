import os
from model import UNet
from dataset import PetDataset, get_pet_dataloader
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
from torch.utils.data import DataLoader

batch_size = 4
base_path = "/Users/venkat/Documents/ERA/transformer/unet/data/"


if __name__ == "__main__":
    # dataset = PetDataset(base_path)
    train_loader, test_loader = get_pet_dataloader()

    # loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    # train_loader = DataLoader(dataset, shuffle=True, **loader_args)
    # val_loader = DataLoader(X_valid, shuffle=False, drop_last=True, **loader_args)
    unet = UNet(in_channels=3, out_channels=1)
    trainer = pl.Trainer(
        max_steps=10_000,
    )
    trainer.fit(
        model=unet,
        train_dataloaders=train_loader,
    )
