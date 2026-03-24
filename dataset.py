import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

INPUT_SIZE = 224


def get_train_transforms() -> transforms.Compose:

    return transforms.Compose([

        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),

        transforms.RandomVerticalFlip(p=0.5),

        transforms.RandomRotation(degrees=90),
 
        transforms.ColorJitter(
            brightness=0.1,
            contrast=0.1,
            saturation=0.1,
            hue=0.0,
        ),

        transforms.ToTensor(),

        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_transforms() -> transforms.Compose:

    return transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class HistoDataset(Dataset):

    def __init__(
        self,
        manifest_path: str | Path,
        split: str,
        transform=None,
    ):
        assert split in {"train", "val", "test"}, \
            f"split must be 'train', 'val', or 'test'. Got: '{split}'"

        self.split     = split
        self.transform = transform


        df = pd.read_csv(manifest_path)
        self.data = df[df["split"] == split].reset_index(drop=True)
        self.image_paths  = self.data["image_path"].tolist()
        self.labels       = self.data["label_index"].tolist()
        self.label_names  = self.data["label"].tolist()

        self.idx_to_label = (
            self.data[["label_index", "label"]]
            .drop_duplicates()
            .set_index("label_index")["label"]
            .to_dict()
        )

    def __len__(self) -> int:
 
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple:

        image_path = self.image_paths[idx]
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            raise IOError(f"Could not load image at index {idx}: {image_path}\n{e}")

    
        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]

        return image, label

    def get_class_weights(self) -> torch.Tensor:
   
        from collections import Counter
        counts  = Counter(self.labels)
        n_total = len(self.labels)
        n_classes = len(counts)

        weights = torch.zeros(n_classes)
        for class_idx, count in counts.items():

            weights[class_idx] = n_total / (n_classes * count)

        return weights


def get_dataloaders(
    manifest_path: str | Path,
    batch_size: int = 32,
    num_workers: int = 2,
) -> dict:
   
    datasets = {
        "train": HistoDataset(manifest_path, split="train", transform=get_train_transforms()),
        "val":   HistoDataset(manifest_path, split="val",   transform=get_val_transforms()),
        "test":  HistoDataset(manifest_path, split="test",  transform=get_val_transforms()),
    }

    dataloaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,

        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
    }


    print("DataLoaders ready:")
    for split, ds in datasets.items():
        dl     = dataloaders[split]
        n_batches = len(dl)
        print(f"  {split:<6}  {len(ds):>5} images  |  {n_batches} batches of {batch_size}")

    return dataloaders, datasets

if __name__ == "__main__":
    import sys
    sys.path.append(".")

    MANIFEST = Path("data/manifest.csv")

    if not MANIFEST.exists():
        print(f"Manifest not found at {MANIFEST}")
        print("Run prepare_dataset.py first.")
        sys.exit(1)

    print("Testing HistoDataset...")
    print()
    train_ds = HistoDataset(MANIFEST, split="train", transform=get_train_transforms())
    print(f"Train dataset size : {len(train_ds)}")

    image, label = train_ds[0]
    print(f"Image tensor shape : {image.shape}")  
    print(f"Image dtype        : {image.dtype}")   
    print(f"Label              : {label}  ({train_ds.idx_to_label[label]})")
    print(f"Pixel value range  : [{image.min():.3f}, {image.max():.3f}]")
 

    print()
    print("Class weights (should all be ~1.0 for balanced dataset):")
    weights = train_ds.get_class_weights()
    for i, w in enumerate(weights):
        print(f"  Class {i}: {w:.4f}")

    print()
    print("Testing DataLoader (one batch)...")
    loader, _ = get_dataloaders(MANIFEST, batch_size=8, num_workers=0)
    images, labels = next(iter(loader["train"]))
    print(f"Batch image shape  : {images.shape}") 
    print(f"Batch labels       : {labels.tolist()}")

    print()
    print("All checks passed. dataset.py is working correctly.")