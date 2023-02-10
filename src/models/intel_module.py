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
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)

        self.log(f"train/loss", loss, prog_bar=True)
        self.log(f"train/acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.hparams.lr,
            momentum=0.9,
            weight_decay=5e-4,
        )
        return {"optimizer": optimizer}


if __name__ == "__main__":
    generate_dataset()
    _ = LitResnet(None,None,None)
