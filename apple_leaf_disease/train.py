import subprocess
from pathlib import Path

import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning.loggers import MLFlowLogger

from apple_leaf_disease.datamodule import AppleLeafDataModule
from apple_leaf_disease.dvc_utils import pull_data_if_missing
from apple_leaf_disease.models.lightning_module import AppleLeafLitModel
from apple_leaf_disease.models.nets import BaselineCNN, build_resnet18


def _freeze_backbone_resnet(net) -> None:
    for name, param in net.named_parameters():
        if 'fc' not in name:
            param.requires_grad = False


def _unfreeze_all(net) -> None:
    for param in net.parameters():
        param.requires_grad = True


def _create_mlflow_logger(cfg, stage_name: str) -> MLFlowLogger:
    logger = MLFlowLogger(
        tracking_uri=cfg.logging.tracking_uri,
        experiment_name=cfg.logging.experiment_name,
        run_name=stage_name,
    )

    git_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()

    params = OmegaConf.to_container(cfg, resolve=True)
    params['git_commit'] = git_commit
    params['stage'] = stage_name

    logger.log_hyperparams(params)
    return logger


def _trainer_common(cfg, logger: MLFlowLogger) -> dict:
    return dict(
        accelerator=cfg.train.accelerator,
        devices=cfg.train.devices,
        precision=cfg.train.precision,
        log_every_n_steps=cfg.train.log_every_n_steps,
        logger=logger,
    )


def _build_lit_model(cfg, net, backbone_lr=None) -> AppleLeafLitModel:
    return AppleLeafLitModel(
        net=net,
        optimizer_cfg=cfg.train.optimizer,
        scheduler_cfg=cfg.train.scheduler,
        threshold=cfg.train.threshold,
        backbone_lr=backbone_lr,
    )


def train(cfg) -> None:
    pull_data_if_missing(
        raw_dir=Path(cfg.data.raw_dir),
        train_csv_name=cfg.data.train_csv_name,
    )

    dm = AppleLeafDataModule(cfg)
    dm.setup()
    num_classes = len(dm.classes)

    if cfg.model.name == 'baseline':
        mlflow_logger = _create_mlflow_logger(cfg, stage_name='baseline')

        net = BaselineCNN(num_classes=num_classes)
        model = _build_lit_model(cfg, net=net, backbone_lr=None)

        trainer = pl.Trainer(
            max_epochs=cfg.train.max_epochs,
            **_trainer_common(cfg, mlflow_logger),
        )
        trainer.fit(
            model,
            train_dataloaders=dm.train_dataloader(),
            val_dataloaders=dm.val_dataloader(),
        )
        return

    if cfg.model.name == 'resnet18':
        net = build_resnet18(num_classes=num_classes, pretrained=bool(cfg.model.pretrained))

        _freeze_backbone_resnet(net)
        logger_stage1 = _create_mlflow_logger(cfg, stage_name='resnet18_stage1_frozen')

        model_stage1 = _build_lit_model(cfg, net=net, backbone_lr=None)

        trainer_stage1 = pl.Trainer(
            max_epochs=int(cfg.model.freeze_backbone_epochs),
            **_trainer_common(cfg, logger_stage1),
        )
        trainer_stage1.fit(
            model_stage1,
            train_dataloaders=dm.train_dataloader(),
            val_dataloaders=dm.val_dataloader(),
        )

        remaining_epochs = int(cfg.train.max_epochs) - int(cfg.model.freeze_backbone_epochs)
        if remaining_epochs > 0:
            _unfreeze_all(net)
            logger_stage2 = _create_mlflow_logger(cfg, stage_name='resnet18_stage2_finetune')

            model_stage2 = _build_lit_model(cfg, net=net, backbone_lr=cfg.model.backbone_lr)

            trainer_stage2 = pl.Trainer(
                max_epochs=int(remaining_epochs),
                **_trainer_common(cfg, logger_stage2),
            )
            trainer_stage2.fit(
                model_stage2,
                train_dataloaders=dm.train_dataloader(),
                val_dataloaders=dm.val_dataloader(),
            )

        return

    raise ValueError(f'Unknown model.name: {cfg.model.name}')
