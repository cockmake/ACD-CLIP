import json
import math
import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from utils import AddGaussianNoise
from .info import CLASS_NAMES, DATA_PATH, DOMAINS


class TextAndImageDataset(Dataset):
    def __init__(
            self,
            data_path: str,
            meta_path: str,
            img_size: int,
    ):
        self.data_path = data_path
        self.img_size = img_size
        self.meta = []
        with open(meta_path, "r") as f:
            for line in f:
                self.meta.append(json.loads(line))

        self.transforms_list = [
            transforms.RandomApply(
                [transforms.RandomRotation(degrees=math.degrees(math.pi / 6))], p=0.5
            ),
            transforms.RandomApply(
                [transforms.RandomAffine(degrees=0, translate=(0.15, 0.15))], p=0.5
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
        ]

        transform_x = [
            AddGaussianNoise(std=1, p=0.7),
            transforms.RandomApply([transforms.ColorJitter(brightness=0.5)], p=0.7),
            transforms.RandomApply([transforms.ColorJitter(contrast=0.5)], p=0.7),
            transforms.RandomApply([transforms.ColorJitter(saturation=0.5)], p=0.7)
        ]
        self.transform_x = transforms.Compose(
            transform_x
            + [
                transforms.Resize((img_size, img_size), InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ],
        )
        self.transform_mask = transforms.Compose(
            [
                transforms.Resize((img_size, img_size), InterpolationMode.NEAREST),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        meta = self.meta[idx]
        data_path = self.data_path
        img_path = os.path.join(data_path, meta["image_path"])
        img = Image.open(img_path).convert("RGB")

        img = self.transform_x(img)
        if meta["label"]:
            mask_path = os.path.join(data_path, meta["mask_path"])
            mask = Image.open(mask_path).convert("L")
            mask = self.transform_mask(mask)
            mask = (mask != 0).float()
        else:
            mask = torch.zeros([1, self.img_size, self.img_size])

        random_transform = transforms.Compose(self.transforms_list)
        transform_tensor = torch.cat([img, mask], dim=0)
        assert transform_tensor.shape[0] == 4
        transform_tensor = random_transform(transform_tensor)
        img = transform_tensor[0:3, :, :]
        mask = transform_tensor[3:4, :, :]

        inputs = {
            "image": img,
            "mask": mask,
            "label": torch.tensor(meta["label"]).to(torch.int64),
            "file_name": meta["image_path"],
            "class_name": meta["class_name"],
        }
        return inputs


class BaseSingleClassDataset(Dataset):
    def __init__(
            self,
            data_path: str,
            meta_path: str,
            img_size: int,
            class_name: str
    ):

        assert class_name is not None, "class_name should be provided"
        self.data_path = data_path
        self.img_size = img_size
        self.meta = []
        with open(meta_path, "r") as f:
            for line in f:
                m = json.loads(line.strip())
                if m["class_name"] == class_name:
                    self.meta.append(m)

        # Define transforms
        self.transform_x = transforms.Compose(
            [
                transforms.Resize((img_size, img_size), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(  # set image / mean metadata from pretrained_cfg if available, or use default
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )
        self.transform_mask = transforms.Compose(
            [
                transforms.Resize((img_size, img_size), Image.NEAREST),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        meta = self.meta[idx]
        img_path = os.path.join(self.data_path, meta["image_path"])
        img = Image.open(img_path).convert("RGB")
        img = self.transform_x(img)
        if meta["label"]:
            mask_path = os.path.join(self.data_path, meta["mask_path"])
            mask = Image.open(mask_path).convert("L")
            mask = self.transform_mask(mask)
            mask = (mask != 0).float()
        else:
            mask = torch.zeros([1, self.img_size, self.img_size])
        inputs = {
            "image": img,
            "mask": mask,
            "label": meta["label"],
            "file_name": meta["image_path"],
            "class_name": meta["class_name"],
        }
        return inputs


def get_text_and_image_dataset(
        dataset_name: str,
        img_size: int,
        stage: str = "train"
):
    if "Med" not in dataset_name:
        assert dataset_name in DATA_PATH, (
            f"Dataset {dataset_name} not found; available datasets: {list(DATA_PATH.keys())}"
        )
    if stage == "train":
        meta_path = os.path.join(
            "./dataset/hub", dataset_name + ".jsonl"
        )
        data_path = DATA_PATH[dataset_name.split("-")[0]]
        dataset = TextAndImageDataset(data_path, meta_path, img_size)
        return dataset
    elif stage == "test":
        meta_path = os.path.join(
            "./dataset/hub", dataset_name + ".jsonl"
        )
        class_names = CLASS_NAMES[dataset_name]
        datasets = {}
        for class_name in class_names:
            image_dataset = BaseSingleClassDataset(
                data_path=DATA_PATH[dataset_name],
                meta_path=meta_path,
                img_size=img_size,
                class_name=class_name
            )
            datasets[class_name] = image_dataset
        return datasets
    else:
        raise ValueError(f"stage {stage} not found; available stages: train, test")
