import os
from model import UNet
from dataset import LoadData, PreprocessData
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
from torch.utils.data import DataLoader

batch_size = 32
base_path = "/Users/venkat/Documents/ERA/transformer/unet/data/"
path1 = base_path + "originals/"
path2 = base_path + "masks/"
img, mask = LoadData(path1, path2)
target_shape_img = [128, 128, 3]
target_shape_mask = [128, 128, 1]
data = PreprocessData(img, mask, target_shape_img, target_shape_mask, path1, path2)
X_train, X_valid = train_test_split(data, test_size=0.2, random_state=123)


if __name__ == "__main__":
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    train_loader = DataLoader(X_train, shuffle=True, **loader_args)
    val_loader = DataLoader(X_valid, shuffle=False, drop_last=True, **loader_args)
    unet = UNet(in_channels=3, out_channels=1)
    trainer = pl.Trainer(
        max_steps=10_000,
    )
    trainer.fit(
        model=unet,
        train_dataloaders=train_loader,
    )
