import torch
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
from typing import Any, Dict, Optional, Tuple, List

import pytorch_lightning as pl
import timm
import torch.nn.functional as F
import torchvision.transforms as T

from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.plugins.environments import LightningEnvironment
from torch.utils.data import DataLoader, Dataset
from torchmetrics import Accuracy
from torchvision.datasets import ImageFolder

import argparse
import glob
import shutil
from torchvision.datasets.utils import extract_archive
from sklearn.model_selection import train_test_split
from collections import Counter
device = "cuda" if torch.cuda.is_available() else "cpu"
accuracy = Accuracy(task="multiclass", num_classes=6).to(device)

class LitResnet(pl.LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        num_classes=6, 
        lr=0.05,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)
        self.model = timm.create_model(
            "resnet18", pretrained=True, num_classes=num_classes
        )

        # loss function
        self.criterion = torch.nn.CrossEntropyLoss()

        # metric objects for calculating and averaging accuracy across batches
        self.train_acc = Accuracy(task="multiclass", num_classes=6)
        self.val_acc = Accuracy(task="multiclass", num_classes=6)
        self.test_acc = Accuracy(task="multiclass", num_classes=6)

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far test accuracy
        self.test_acc_best = MaxMetric()

    def forward(self, x):
        out = self.model(x)
        return F.log_softmax(out, dim=1)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        self.test_acc_best.reset()
    
    def model_step(self, batch: Any):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def evaluate(self, batch, stage=None):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)

        if stage:
            self.log(f"{stage}/loss", loss, prog_bar=True)
            self.log(f"{stage}/acc", acc, prog_bar=True)

    def training_step(self, batch, batch_idx):
        loss, preds, targets = self.model_step(batch)
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)
        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        #self.log(f"train/loss", loss, prog_bar=True)
        #self.log(f"train/acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def validation_epoch_end(self, outputs: List[Any]):
        acc = self.val_acc.compute()  # get current val acc
        #self.val_acc_best(acc)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        #self.log("val/acc_best", self.val_acc_best.compute(), prog_bar=True)
                

    def test_step(self, batch, batch_idx):
        loss, preds, targets = self.model_step(batch)
        self.evaluate(batch, "test")
        self.test_loss(loss)
        self.test_acc(preds, targets)
        #self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        #self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)
        acc = self.test_acc.compute()  # get current val acc
        self.test_acc_best(acc)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("test/acc_best", self.test_acc_best.compute(), prog_bar=True)
    
    def configure_optimizers(self):
        # optimizer = torch.optim.SGD(
        #     self.parameters(),
        #     lr=self.hparams.lr,
        #     momentum=0.9,
        #     weight_decay=5e-4,
        # )
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    generate_dataset()
    _ = LitResnet(None,None,None)
