from typing import Any, Dict, Optional, Tuple

import json
import os
from datetime import datetime
from pathlib import Path

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
#from torchvision.datasets import MNIST
from torchvision.transforms import transforms

import pytorch_lightning as pl
import timm
import torch.nn.functional as F
import torchvision.transforms as T

from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.plugins.environments import LightningEnvironment
from torchmetrics import Accuracy
from torchvision.datasets import ImageFolder

import argparse
import glob
import shutil
from torchvision.datasets.utils import extract_archive
from sklearn.model_selection import train_test_split
from collections import Counter

def write_dataset(image_paths, output_dir):
    for img_path in image_paths:
        Path(output_dir / img_path.parent.stem).mkdir(parents=True, exist_ok=True)
        shutil.copyfile(img_path, output_dir / img_path.parent.stem / img_path.name)

def generate_dataset():
    dataset_extracted = Path("data/intel-image-classification")
    dataset_extracted.mkdir(parents=True, exist_ok=True)

    # split dataset and save to their directories
    print(f":: Extracting Zip {dataset_zip} to {dataset_extracted}")
    extract_archive(from_path=dataset_zip, to_path=dataset_extracted)

    ds = list((dataset_extracted / "seg_train" / "seg_train").glob("*/*"))
    ds += list((dataset_extracted / "seg_test" / "seg_test").glob("*/*"))
    d_pred = list((dataset_extracted / "seg_pred" / "seg_pred").glob("*/"))

    labels = [x.parent.stem for x in ds]
    print(":: Dataset Class Counts: ", Counter(labels))

    d_train, d_test = train_test_split(ds, test_size=0.2, stratify=labels)
    d_test, d_val = train_test_split(
        d_test, test_size=0.5, stratify=[x.parent.stem for x in d_test]
    )

    print("\t:: Train Dataset Class Counts: ", Counter(x.parent.stem for x in d_train))
    print("\t:: Test Dataset Class Counts: ", Counter(x.parent.stem for x in d_test))
    print("\t:: Val Dataset Class Counts: ", Counter(x.parent.stem for x in d_val))
    print("\t:: Total validation images", len(d_pred))
    dataset_path=Path("data")
    for path in ["train", "test", "val","preds"]:
        output_dir = dataset_path / path
        print(f"\t:: Creating Directory {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)

    print(":: Writing Datasets")
    write_dataset(d_train, Path("data/train"))
    write_dataset(d_test, Path("data/test"))
    write_dataset(d_val, Path("data/val"))
    write_dataset(d_pred, Path("data/preds"))

dataset_zip = Path("data/intel.zip")
class IntelImgClfDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/",
        batch_size: int = 256,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_dir = Path(data_dir)

        # data transformations
        self.transforms = T.Compose(
            [
                T.ToTensor(),
                T.Resize((224, 224)),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        self.data_train: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        generate_dataset()


    @property
    def num_classes(self):
        return len(self.data_train.classes)

    @property
    def classes(self):
        return self.data_train.classes

    def prepare_data(self):
        """Download data if needed.
        Do not use it to assign state (self.x = y).
        """
        pass

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.
        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_test:
            trainset = ImageFolder(self.data_dir / "train", transform=self.transforms)
            testset = ImageFolder(self.data_dir / "test", transform=self.transforms)
            valset = ImageFolder(self.data_dir / "val", transform=self.transforms)

            self.data_train, self.data_test, self.data_val = trainset, testset, valset

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass


def train_and_evaluate(model, datamodule, sm_training_env, output_dir):
    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir=ml_root / "output" / "tensorboard" / sm_training_env["job_name"]
    )
    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="auto",
        logger=[tb_logger],
        callbacks=[TQDMProgressBar(refresh_rate=10)],
    )

    trainer.fit(model, datamodule)
    trainer.test(model, datamodule)

    idx_to_class = {k: v for v, k in datamodule.data_train.class_to_idx.items()}
    model.idx_to_class = idx_to_class

    # per class accuracy
    confusion_matrix = torch.zeros(datamodule.num_classes, datamodule.num_classes)
    with torch.no_grad():
        for i, (images, targets) in enumerate(datamodule.test_dataloader()):
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            for t, p in zip(targets.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

    acc_per_class = {
        idx_to_class[idx]: val.item() * 100
        for idx, val in enumerate(confusion_matrix.diag() / confusion_matrix.sum(1))
    }
    print(acc_per_class)

    with open(output_dir / "accuracy_per_class.json", "w") as outfile:
        json.dump(acc_per_class, outfile)
    return trainer

if __name__ == "__main__":
    _ = IntelImgClfDataModule()
