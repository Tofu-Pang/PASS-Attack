import csv
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor


class NIPSDataset(torch.utils.data.Dataset):
    def __init__(self, root):
        self.root = Path(root)
        csv_path = self.root / "images.csv"
        image_dir = self.root / "images"
        if not csv_path.is_file():
            raise FileNotFoundError(csv_path)
        if not image_dir.is_dir():
            raise FileNotFoundError(image_dir)
        records = []
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"ImageId", "TrueLabel", "TargetClass"}
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise ValueError(f"CSV columns must contain {sorted(required)}")
            for row in reader:
                path = image_dir / f"{row['ImageId']}.png"
                if not path.is_file():
                    raise FileNotFoundError(path)
                true_label = int(row["TrueLabel"])
                target_label = int(row["TargetClass"])
                if not 1 <= true_label <= 1000 or not 1 <= target_label <= 1000:
                    raise ValueError(f"Invalid ImageNet label for {row['ImageId']}")
                if true_label == target_label:
                    raise ValueError(f"Target label matches true label for {row['ImageId']}")
                records.append((path, true_label - 1, target_label - 1))
        if not records:
            raise ValueError("Dataset is empty")
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        path, true_label, target_label = self.records[index]
        with Image.open(path) as image:
            tensor = pil_to_tensor(image.convert("RGB")).float().div_(255.0)
        return tensor, true_label, target_label
