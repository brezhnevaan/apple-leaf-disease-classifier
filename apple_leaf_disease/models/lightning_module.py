from collections.abc import Mapping
from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics.classification import MultilabelF1Score


def _require(cfg: Mapping[str, Any], key: str) -> Any:
    if key not in cfg:
        raise KeyError(f"Missing required config key: '{key}'")
    return cfg[key]


class AppleLeafLitModel(pl.LightningModule):
    def __init__(
        self,
        net: nn.Module,
        optimizer_cfg: Mapping[str, Any],
        scheduler_cfg: Mapping[str, Any] | None,
        threshold: float,
        backbone_lr: float | None = None,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(ignore=['net', 'optimizer_cfg', 'scheduler_cfg'])

        self.net = net
        self.criterion = nn.BCEWithLogitsLoss()

        self.optimizer_cfg = dict(optimizer_cfg)
        self.scheduler_cfg = dict(scheduler_cfg) if scheduler_cfg is not None else None

        self.threshold = float(threshold)
        self.backbone_lr = float(backbone_lr) if backbone_lr is not None else None

        self.f1_macro: MultilabelF1Score | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def training_step(self, batch, batch_idx: int):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        self.log('train/loss_step', loss, prog_bar=True, on_step=True, on_epoch=False)

        return loss

    def validation_step(self, batch, batch_idx: int):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        preds = torch.sigmoid(logits)

        if self.f1_macro is None:
            num_classes = int(preds.shape[1])
            self.f1_macro = MultilabelF1Score(
                num_labels=num_classes,
                average='macro',
                threshold=self.threshold,
            ).to(self.device)

        f1 = self.f1_macro(preds, y.int())

        self.log('val/loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val/f1_macro', f1, prog_bar=True, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        # --- optimizer config (strict) ---
        opt_name = str(_require(self.optimizer_cfg, 'name')).lower()
        lr = float(_require(self.optimizer_cfg, 'lr'))
        weight_decay = float(_require(self.optimizer_cfg, 'weight_decay'))

        if opt_name != 'adamw':
            raise ValueError(f"Unsupported optimizer.name: {opt_name}. Expected 'adamw'.")

        if self.backbone_lr is None:
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=lr,
                weight_decay=weight_decay,
            )
        else:
            backbone_params = []
            head_params = []

            for name, param in self.net.named_parameters():
                if not param.requires_grad:
                    continue
                if 'fc' in name or 'classifier' in name:
                    head_params.append(param)
                else:
                    backbone_params.append(param)

            if len(head_params) == 0:
                raise RuntimeError('Could not find head parameters (fc/classifier).')
            if len(backbone_params) == 0:
                raise RuntimeError('Could not find backbone parameters.')

            optimizer = torch.optim.AdamW(
                [
                    {'params': backbone_params, 'lr': float(self.backbone_lr)},
                    {'params': head_params, 'lr': lr},
                ],
                weight_decay=weight_decay,
            )

        # --- scheduler config (strict optional) ---
        if self.scheduler_cfg is None:
            return optimizer

        sch_name = str(_require(self.scheduler_cfg, 'name')).lower()
        if sch_name != 'reduce_on_plateau':
            raise ValueError(
                f"Unsupported scheduler.name: {sch_name}. Expected 'reduce_on_plateau'."
            )

        monitor = str(_require(self.scheduler_cfg, 'monitor'))
        mode = str(_require(self.scheduler_cfg, 'mode'))
        factor = float(_require(self.scheduler_cfg, 'factor'))
        patience = int(_require(self.scheduler_cfg, 'patience'))
        min_lr = float(_require(self.scheduler_cfg, 'min_lr'))

        scheduler = ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=factor,
            patience=patience,
            min_lr=min_lr,
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': monitor,
                'interval': 'epoch',
                'frequency': 1,
            },
        }
