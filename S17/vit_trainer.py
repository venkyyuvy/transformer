import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import pytorch_lightning as pl
# Create a random tensor with same shape as a single image
from vit import ViT
# Setup the optimizer to optimize our ViT model parameters using hyperparameters from the ViT paper 
# Example of creating the class embedding and expanding over a batch dimension

device = "mps"
batch_size = 32
num_workers = 4
IMG_SIZE = 224
image_path = "./pizza_steak_sushi"
train_dir = image_path + "/train"
test_dir = image_path + "/test"

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])           


  # Use ImageFolder to create dataset(s)
train_data = datasets.ImageFolder(train_dir, transform=transform)
test_data = datasets.ImageFolder(test_dir, transform=transform)

# Get class names
class_names = train_data.classes

# Turn images into data loaders
train_dataloader = DataLoader(
  train_data,
  batch_size=batch_size,
  shuffle=True,
  num_workers=num_workers,
  pin_memory=True,
)
val_dataloader = DataLoader(
  test_data,
  batch_size=batch_size,
  shuffle=False,
  num_workers=num_workers,
  pin_memory=True,
)

# results = engine.train(model=vit,
#                        train_dataloader=train_dataloader,
#                        test_dataloader=test_dataloader,
#                        optimizer=optimizer,
#                        loss_fn=loss_fn,
#                        epochs=2,
#                        device=device)


class LitViT(ViT, pl.LightningModule):
    def __init__(self, num_classes=len(class_names)):
        super().__init__(num_classes=num_classes)

    def training_step(self, batch, batch_idx):
        X, y = batch
        y_pred = self(X)
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(y_pred, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            params=self.parameters(), 
            lr=3e-3,
            betas=(0.9, 0.999),
            weight_decay=0.3
        ) 
        return optimizer


# model
if __name__ == '__main__':
    vit = LitViT()

    # train model
    trainer = pl.Trainer(
        max_epochs=10,
    )
    trainer.fit(
        model=vit,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )
