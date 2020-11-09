import os
from pathlib import Path
from warnings import warn

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import download_and_extract_archive

from pytorch_lightning import LightningDataModule

data_dir = Path('data/imagewoof2/')

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


class WoofDataModule(LightningDataModule):
    '''Woof dataset Datamodule'''
    name = 'woof'

    def __init__(
            self,
            data_dir: str = data_dir,
            image_size: int = 192,
            num_workers: int = 4,
            batch_size: int = 32,
            # *args,
            # **kwargs,
    ):
        # '''
        # Args: ...
        # '''
        # super().__init__(*args, **kwargs)
        super().__init__()
        self.image_size = image_size
        self.dims = (3, self.image_size, self.image_size)
        self.data_dir = Path(data_dir)
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.num_classes = 10
        self.train_img_scale = (0.35, 1)

    def _verify_data(self, data_dir, split):
        dataset_len = {'train': 9025, 'val': 3929}
        num_classes = 10
        split_path = Path(data_dir, split)
        if not data_dir.exists():
            raise FileNotFoundError(f"Directory {split_path} not exist")
        classes_dirs = [dir_entry for dir_entry in os.scandir(split_path)
                        if dir_entry.is_dir()]
        if num_classes != len(classes_dirs):
            warn(f"{num_classes} dirs expected, but has {len(classes_dirs)} dirs.")

        num_samples = 0
        for dir_entry in classes_dirs:
            num_samples += len([fn for fn in os.scandir(dir_entry)
                                if fn.is_file()])
        if num_samples != dataset_len[split]:
            warn(f"Len of {dataset_len[split]} items expected at {split} dirs, \
                but has {num_samples} item.")

    def prepare_data(self):
        """
        """
        imafewoof_url = 'https://s3.amazonaws.com/fast-ai-imageclas/imagewoof2.tgz'
        # imafewoof_url = 'https://s3.amazonaws.com/fast-ai-imageclas/imagewoof2-160.tgz'
        if not self.data_dir.exists():
            os.makedirs(self.data_dir)
            download_and_extract_archive(imafewoof_url, self.data_dir.parent, remove_finished=False)
            # raise FileNotFoundError(f"Directory {self.data_dir} not exist")
        print('Call verify')
        self._verify_data(self.data_dir, 'train')
        self._verify_data(self.data_dir, 'val')

    def setup(self, stage=None):
        self._verify_data(self.data_dir, 'train')
        self._verify_data(self.data_dir, 'val')
        train_transforms = self.train_transform() if self.train_transforms is None else self.train_transforms
        self.train_dataset = ImageFolder(root=Path(self.data_dir, 'train'), transform=train_transforms)
        val_transforms = self.val_transform() if self.val_transforms is None else self.val_transforms
        self.val_dataset = ImageFolder(root=Path(self.data_dir, 'val'), transform=val_transforms)

    def train_dataloader(self):
        """
        Uses the train split of woof
        """

        loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True
        )
        return loader

    def val_dataloader(self):
        """
        Uses the valid part of the woof
        """

        loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        return loader

    def train_transform(self):
        """
        The standard imagenet transforms

        .. code-block:: python

            transform_lib.Compose([
                transform_lib.RandomResizedCrop(self.image_size),
                transform_lib.RandomHorizontalFlip(),
                transform_lib.ToTensor(),
                transform_lib.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])

        """
        preprocessing = transforms.Compose([
            transforms.RandomResizedCrop(self.image_size, scale=self.train_img_scale),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # imagenet_normalization(),
            normalize,
        ])

        return preprocessing

    def val_transform(self):
        """
        The standard imagenet transforms for validation

        .. code-block:: python

            transform_lib.Compose([
                transform_lib.Resize(self.image_size + 32),
                transform_lib.CenterCrop(self.image_size),
                transform_lib.ToTensor(),
                transform_lib.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])

        """

        preprocessing = transforms.Compose([
            transforms.Resize(self.image_size + 32),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor(),
            # imagenet_normalization(),
            normalize,
        ])
        return preprocessing
