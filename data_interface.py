import os
import torch
import pydicom
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import torchvision.transforms as T


class DicomDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        """
        Args:
            csv_file (str): CSV 文件，包含相对路径和标签
            transform (callable, optional): 图像变换
        """
        import pandas as pd
        self.data_info = pd.read_csv(csv_file, header=None)
        self.transform = transform

        # 标签映射
        self.label_map = {"good": 0, "bad": 1}

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        img_path = self.data_info.iloc[idx, 0]
        label_str = self.data_info.iloc[idx, 1]
        label = self.label_map[label_str]

        dcm = pydicom.dcmread(img_path, force=True)
        if not hasattr(dcm, "PhotometricInterpretation") or dcm.PhotometricInterpretation == "":
            dcm.PhotometricInterpretation = "MONOCHROME2"

        try:
            image = dcm.pixel_array.astype(np.float32)
        except:
            rows, cols = int(dcm.Rows), int(dcm.Columns)
            image = np.frombuffer(dcm.PixelData, dtype=np.uint16).reshape(rows, cols).astype(np.float32)

        # 归一化
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        image = torch.from_numpy(image).unsqueeze(0)  # (1,H,W)

        if self.transform:
            image = self.transform(image)

        # 扩展为 3 通道
        image = image.repeat(3, 1, 1)  # (3,H,W)

        return image, label


class DInterface(pl.LightningDataModule):
    def __init__(self, csv_file, batch_size=8, num_workers=4, aug_type="default", **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.csv_file = csv_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.aug_type = aug_type

        # 默认 resize
        self.base_transform = T.Compose([
            T.Resize((256, 256))
        ])

        # 数据增强 pipeline
        self.aug_transforms = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(15),
            T.GaussianBlur(3),
            T.Resize((256, 256))
        ])

    def setup(self, stage=None):
        transform = self.base_transform if self.aug_type == "default" else self.aug_transforms
        dataset = DicomDataset(self.csv_file, transform=transform)

        n_total = len(dataset)
        n_train = int(0.8 * n_total)
        n_val = n_total - n_train
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(dataset, [n_train, n_val])

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers)
