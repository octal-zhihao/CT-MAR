from pytorch_lightning import LightningModule
import torch.nn as nn
import torch
from torchmetrics.classification import (
    Accuracy,
    Precision,
    Recall,
    F1Score
)
from .classifier import Model4Classifier
from .AlexNet import AlexNet

class MInterface(LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()  
        
        # 初始化指标
        num_classes = kwargs["class_num"]
        
        self.criterion = nn.CrossEntropyLoss()
        metric_args = {
            "task": "multiclass",
            "num_classes": num_classes,
            "average": "weighted"  # 可选'micro'/'macro'
        }
        self.train_metrics = nn.ModuleDict({
            "micro_acc": Accuracy(
                task = "multiclass",
                num_classes = num_classes,
                average = "micro"
            ),
            "acc": Accuracy(**metric_args),
            "precision": Precision(**metric_args),
            "recall": Recall(**metric_args),
            "f1": F1Score(**metric_args)
        })
        
        self.val_metrics = nn.ModuleDict({
            "micro_acc": Accuracy(
                task = "multiclass",
                num_classes = num_classes,
                average = "micro"
            ),
            "acc": Accuracy(**metric_args),
            "precision": Precision(**metric_args),
            "recall": Recall(**metric_args),
            "f1": F1Score(**metric_args)
        })
        
        self.test_metrics = nn.ModuleDict({
            "micro_acc": Accuracy(
                task = "multiclass",
                num_classes = num_classes,
                average = "micro"
            ),
            "acc": Accuracy(**metric_args),
            "precision": Precision(**metric_args),
            "recall": Recall(**metric_args),
            "f1": F1Score(**metric_args)
        })
        if self.hparams.model_name == "ResNet":
            self.model = Model4Classifier(**kwargs)
        if self.hparams.model_name == "EfficientNetV2":
            self.model = Model4Classifier(**kwargs)
        if self.hparams.model_name == "SwinTransformer":
            self.model = Model4Classifier(**kwargs)
        if self.hparams.model_name == "ConvNeXt":
            self.model = Model4Classifier(**kwargs)
        if self.hparams.model_name == "VGG":
            self.model = Model4Classifier(**kwargs)
        
        if self.hparams.model_name == "AlexNet":
            self.model = AlexNet(num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch, stage):
        x, y = batch
        # 处理混合标签的损失计算
        if stage == "train" and y.dim() == 2:  # one-hot格式
            y_hat = self(x)
            loss = torch.sum(-y * torch.log_softmax(y_hat, dim=1), dim=1).mean()
            y_pred = torch.argmax(y_hat, dim=1)
            y_true = torch.argmax(y, dim=1)  # 用于指标计算
        else:
            y_hat = self(x)
            loss = self.criterion(y_hat, y)
            y_pred = torch.argmax(y_hat, dim=1)
            y_true = y
        
        metrics = getattr(self, f"{stage}_metrics")
        # 先记录loss和acc到进度条
        self.log(
            f"{stage}_loss", 
            loss, 
            prog_bar=(stage != "test"), 
            on_step=False,
            on_epoch=True
        )
        self.log(
            f"{stage}_acc",
            metrics["acc"](y_pred, y_true),
            prog_bar=(stage != "test"),
            on_step=False,
            on_epoch=True
        )
        
        # 其他指标仅记录到日志不显示在进度条
        self.log_dict(
            {
                f"{stage}_micro_acc": metrics["micro_acc"](y_pred, y_true),
                f"{stage}_precision": metrics["precision"](y_pred, y_true),
                f"{stage}_recall": metrics["recall"](y_pred, y_true),
                f"{stage}_f1": metrics["f1"](y_pred, y_true)
            },
            prog_bar=False,
            on_step=False,
            on_epoch=True
        )
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW (
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        if self.hparams.lr_scheduler:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=int(self.hparams.epochs*0.5),
                eta_min=1e-5
            )
            return [optimizer], [scheduler]
        
        return optimizer