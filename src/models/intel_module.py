import torch
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
from typing import Any, Dict, Optional, Tuple, List
import torchvision
import pytorch_lightning as pl
import timm
import torch.nn.functional as F
import torchvision.transforms as T

from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.plugins.environments import LightningEnvironment
from torch.utils.data import DataLoader, Dataset
from torchmetrics import Accuracy
from torchmetrics import F1Score, Precision, Recall, ConfusionMatrix, MaxMetric, MeanMetric
from torchvision.datasets import ImageFolder
from PIL import Image
import pandas as pd
import seaborn as sn
import io
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
accuracy = Accuracy(task="multiclass", num_classes=6).to(device)
precision=Precision(task='multiclass',average='macro',num_classes=6).to(device)
recall = Recall(task="multiclass", average='macro', num_classes=6).to(device)
confmat = ConfusionMatrix(task="multiclass", num_classes=6).to(device)

class IntHandler:
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        text = plt.matplotlib.text.Text(x0, y0, str(orig_handle))
        handlebox.add_artist(text)
        return text
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

        self.num_classes = num_classes
        self.save_hyperparameters()        
        self.model = timm.create_model(
            "resnet18", pretrained=True, num_classes=num_classes
        )

        # loss function
        self.criterion = torch.nn.CrossEntropyLoss()

        # metric objects for calculating and averaging accuracy across batches
        self.train_acc = Accuracy(task="multiclass", num_classes=6)
        self.val_acc = Accuracy(task="multiclass", num_classes=6)
        self.test_acc = Accuracy(task="multiclass", num_classes=6)

        # some other metrics to be logged
        self.f1_score = F1Score(task="multiclass", num_classes=self.num_classes)
        self.precision_score = Precision(task="multiclass", average='macro', num_classes=self.num_classes)
        self.recall_score = Recall(task="multiclass", average='macro', num_classes=self.num_classes)

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        #self.val_acc_best = MaxMetric()

        # for tracking best so far test accuracy
        self.test_acc_best = MaxMetric()
    
    def forward(self, x):
        #out = self.model(x)
        #return F.log_softmax(out, dim=1)
        return self.net(x)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        #self.val_acc_best.reset()
        self.test_acc_best.reset()
        pass
    
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

        # update and log metrics
        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": targets}
    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    def validation_step(self, batch, batch_idx):
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.f1_score(preds, targets)
        self.precision_score(preds, targets)
        self.recall_score(preds, targets)
        self.log("val/loss", self.val_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=True, on_epoch=True, prog_bar=False)
        self.log("val/f1", self.val_acc, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/precision", self.precision_score, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/recall", self.recall_score, on_step=False, on_epoch=True, prog_bar=False)
        return {"loss": loss, "preds": preds, "targets": targets}

    # def validation_epoch_end(self, outputs: List[Any]):
    #     #acc = self.val_acc.compute()  # get current val acc
    #     #self.val_acc_best(acc)  # update best so far val acc
    #     # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
    #     # otherwise metric would be reset by lightning after each epoch
    #     #self.log("val/acc_best", self.val_acc_best.compute(), prog_bar=True)
    #     pass

    def validation_epoch_end(self, outs: List[Any]):
        tb = self.logger.experiment  # noqa

        outputs = torch.cat([tmp['preds'] for tmp in outs])
        labels = torch.cat([tmp['targets'] for tmp in outs])

        confusion = ConfusionMatrix(task="multiclass", num_classes=self.num_classes).to(device)
        confusion(outputs, labels)
        computed_confusion = confusion.compute().detach().cpu().numpy().astype(int)

        # confusion matrix
        df_cm = pd.DataFrame(
            computed_confusion,
            index=[0, 1, 2, 3, 4, 5],
            columns=['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street'],
        )

        fig, ax = plt.subplots(figsize=(10, 5))
        fig.subplots_adjust(left=0.05, right=.65)
        sn.set(font_scale=1.2)
        sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, fmt='d', ax=ax)
        ax.legend(
            [0, 1, 2, 3, 4, 5],
            ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street'],
            handler_map={int: IntHandler()},
            loc='upper left',
            bbox_to_anchor=(1.2, 1)
        )
        buf = io.BytesIO()

        plt.savefig(buf, format='jpeg', bbox_inches='tight')
        buf.seek(0)
        im = Image.open(buf)
        im = torchvision.transforms.ToTensor()(im)
        tb.add_image("val_confusion_matrix", im, global_step=self.current_epoch)
                

    def test_step(self, batch, batch_idx):
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.test_acc(preds, targets)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)
        acc = self.test_acc.compute()  # get current val acc
        self.test_acc_best(acc)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("test/acc_best", self.test_acc_best.compute(), prog_bar=True)
        return {"loss": loss, "preds": preds, "targets": targets}
    
    def test_epoch_end(self, outputs: List[Any]):
        #acc = self.test_acc.compute()  # get current val acc
        #self.test_acc_best(acc)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        #self.log("test/acc_best", self.test_acc_best.compute(), prog_bar=True)
        pass

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
                    "monitor": "train/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    #generate_dataset()
    _ = LitResnet(None,None,None)
