from model import UNet
from dataset import get_pet_dataloader
import pytorch_lightning as pl

batch_size = 4


if __name__ == "__main__":
    # dataset = PetDataset(base_path)
    train_loader, test_loader = get_pet_dataloader(batch_size=batch_size)

    # loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    # train_loader = DataLoader(dataset, shuffle=True, **loader_args)
    # val_loader = DataLoader(X_valid, shuffle=False, drop_last=True, **loader_args)
    unet = UNet(
        loss_fn="ce",
        contract_method="sc",
        expand_method="tr"
    )
    trainer = pl.Trainer(
        max_steps=10_000,
        strategy='ddp_find_unused_parameters_true',
    )
    trainer.fit(
        model=unet,
        train_dataloaders=train_loader,

    )
