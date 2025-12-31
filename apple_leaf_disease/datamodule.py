from pathlib import Path

import pandas as pd
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from apple_leaf_disease.dataset import AppleLeafDataset


class AppleLeafDataModule:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.raw_dir = Path(cfg.data.raw_dir)
        self.csv_path = self.raw_dir / cfg.data.train_csv_name
        self.images_dir = self.raw_dir / cfg.data.train_images_dir

        self.batch_size = int(cfg.data.batch_size)
        self.num_workers = int(cfg.data.num_workers)
        self.seed = int(cfg.data.seed)

        self.val_size = float(cfg.data.val_size)
        self.test_size = float(cfg.data.test_size)

        self.pin_memory = bool(getattr(cfg.data, 'pin_memory', False))
        self.persistent_workers = bool(getattr(cfg.data, 'persistent_workers', False))

        self.classes: list[str] = []
        self.train_ds: AppleLeafDataset | None = None
        self.val_ds: AppleLeafDataset | None = None
        self.test_ds: AppleLeafDataset | None = None

    def _build_transforms(self):
        aug = self.cfg.augment

        mean = list(aug.normalize.mean)
        std = list(aug.normalize.std)

        train_transforms = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    size=int(aug.img_size),
                    scale=(float(aug.train.crop_scale_min), float(aug.train.crop_scale_max)),
                ),
                transforms.RandomHorizontalFlip(p=float(aug.train.hflip_p)),
                transforms.RandomRotation(degrees=float(aug.train.rotation_deg)),
                transforms.ColorJitter(
                    brightness=float(aug.train.jitter.brightness),
                    contrast=float(aug.train.jitter.contrast),
                    saturation=float(aug.train.jitter.saturation),
                    hue=float(aug.train.jitter.hue),
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

        val_transforms = transforms.Compose(
            [
                transforms.Resize(int(aug.resize_size)),
                transforms.CenterCrop(int(aug.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

        return train_transforms, val_transforms

    def setup(self) -> None:
        df = pd.read_csv(self.csv_path)

        self.classes = sorted({lab for s in df['labels'] for lab in str(s).split(' ') if lab})

        df_trainval, df_test = train_test_split(
            df,
            test_size=self.test_size,
            random_state=self.seed,
            shuffle=True,
        )

        val_size_rel = self.val_size / (1.0 - self.test_size)

        df_train, df_val = train_test_split(
            df_trainval,
            test_size=val_size_rel,
            random_state=self.seed,
            shuffle=True,
        )

        train_tfms, val_tfms = self._build_transforms()

        self.train_ds = AppleLeafDataset(
            csv_path=self.csv_path,
            images_dir=self.images_dir,
            classes=self.classes,
            transform=train_tfms,
        )
        self.val_ds = AppleLeafDataset(
            csv_path=self.csv_path,
            images_dir=self.images_dir,
            classes=self.classes,
            transform=val_tfms,
        )
        self.test_ds = AppleLeafDataset(
            csv_path=self.csv_path,
            images_dir=self.images_dir,
            classes=self.classes,
            transform=val_tfms,
        )

        self.train_ds.df = df_train.reset_index(drop=True)
        self.val_ds.df = df_val.reset_index(drop=True)
        self.test_ds.df = df_test.reset_index(drop=True)

    def _dl_kwargs(self) -> dict:
        persistent_workers = self.persistent_workers and self.num_workers > 0
        return dict(
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=persistent_workers,
        )

    def train_dataloader(self) -> DataLoader:
        assert self.train_ds is not None
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            **self._dl_kwargs(),
        )

    def val_dataloader(self) -> DataLoader:
        assert self.val_ds is not None
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            **self._dl_kwargs(),
        )

    def test_dataloader(self) -> DataLoader:
        assert self.test_ds is not None
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            **self._dl_kwargs(),
        )
