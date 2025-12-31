from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as transforms


class AppleLeafDataset(Dataset):
    def __init__(
        self,
        csv_path: Path,
        images_dir: Path,
        classes: list[str],
        transform: transforms.Compose | None = None,
    ) -> None:
        self.df = pd.read_csv(csv_path)
        self.images_dir = images_dir
        self.classes = classes
        self.class_to_idx = {c: i for i, c in enumerate(classes)}

        if transform is None:
            raise ValueError('transform must be provided (build it from cfg in DataModule)')

        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]

        image_id = row['image']
        labels_raw = row.get('labels', '')
        labels = str(labels_raw).split(' ') if pd.notna(labels_raw) else []

        image_name = str(image_id)
        if not image_name.lower().endswith('.jpg'):
            image_name = f'{image_name}.jpg'
        image_path = self.images_dir / image_name

        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        target = torch.zeros(len(self.classes), dtype=torch.float32)
        for label in labels:
            if label:
                target[self.class_to_idx[label]] = 1.0

        return image, target
