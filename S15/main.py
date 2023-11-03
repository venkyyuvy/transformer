from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl

from datasets import load_dataset

from dataset import BilingualDataset
from config import get_config
from model import build_transfomer
from trainer import run_batch_validation, get_or_build_tokenizer


class TranslationDataModule(pl.LightningModule):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.ds_raw = load_dataset(
            "opus_books",
            f"{config['lang_src']}-{config['lang_tgt']}",
            split='train'
        )

        self.tokenizer_src = get_or_build_tokenizer(
            self.config, self.ds_raw, config['lang_src'])
        self.tokenizer_tgt = get_or_build_tokenizer(
            self.config, self.ds_raw, config['lang_tgt'])

    def setup(self, stage=None):

        train_ds_size = int(0.9 * len(self.ds_raw))
        val_ds_size = len(self.ds_raw) - train_ds_size
        train_ds_raw, val_ds_raw = random_split(
            self.ds_raw, 
            [train_ds_size, val_ds_size]
        )

        self.train_ds = BilingualDataset(
            train_ds_raw, 
            self.tokenizer_src, 
            self.tokenizer_tgt, 
            config['lang_src'], 
            config['lang_tgt'],
            config['seq_len']
        )
        self.val_ds = BilingualDataset(
            val_ds_raw,
            self.tokenizer_src,
            self.tokenizer_tgt, 
            config['lang_src'],
            config['lang_tgt'],
            config['seq_len']
        )


    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_ds,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=config['num_workers'],
            drop_last=False
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=config['num_workers'],
            drop_last=False
        )


class PlTransformers(pl.LightningModule):
    def __init__(
            self, 
            config, 
            model,
            tokenizer_src,
            tokenizer_tgt
        ) -> None:
        self.config = config
        super().__init__()
        Path(self.config['model_folder']).mkdir(parents=True, exist_ok=True)
        self.model = model
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=config['lr'], eps=1e-9)
        return optimizer

    def criterion(self, out, y):
        loss_fn =  nn.CrossEntropyLoss(
            ignore_index=self.tokenizer_tgt.token_to_id('[PAD]'),
            label_smoothing=0.1
        )

        return loss_fn(
            out.view(-1, self.tokenizer_tgt.get_vocab_size()),
            y.view(-1)
        )

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        lr = opt.param_groups[0]['lr']
        self.log('learning_rate', lr)
        encoder_input = batch['encoder_input']
        decoder_input = batch['decoder_input']
        encoder_mask = batch['encoder_mask']
        decoder_mask = batch['decoder_mask']

        encoder_output = self.model.encode(encoder_input, encoder_mask)
        decoder_output = self.model.decode(
            encoder_output, encoder_mask,
            decoder_input, decoder_mask
        )
        proj_output = self.model.project(decoder_output)
        # (B, seq_len, vocab_size)

        label = batch['label']
        loss = self.criterion(proj_output, label)
        self.log(
            "train_loss", loss, prog_bar=True,
            logger=True, on_step=True, on_epoch=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        run_batch_validation(
            self,
            batch,
            self.tokenizer_tgt,
            config['seq_len'],
            batch_idx,
            self.log
            )


if __name__ == '__main__':
    config = get_config()

    trainer = pl.Trainer(
        accelerator=config['DEVICE'], devices=-1,
        strategy=config['STRATEGY'],
        max_epochs = 30,
        enable_progress_bar = True,
        log_every_n_steps = 1,
        precision='16-mixed',
        limit_train_batches=0.05,
        check_val_every_n_epoch=2,
        num_sanity_val_steps=0,
        limit_val_batches=10,
        # limit_val_batches=0.02,
        # check_val_every_n_epoch=10,
        # limit_test_batches=0.01,
        # detect_anomaly=True
    )


    data_module = TranslationDataModule(config)
    model = build_transfomer(
        data_module.tokenizer_src.get_vocab_size(),
        data_module.tokenizer_tgt.get_vocab_size(),
        config["seq_len"],
        config["seq_len"], 
        d_model=config['d_model']
    )
 

    transformer = PlTransformers(
        config, 
        model,
        data_module.tokenizer_src,
        data_module.tokenizer_tgt
    )

    trainer.fit(transformer, datamodule=data_module)

