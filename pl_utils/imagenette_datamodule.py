import os
from pathlib import Path
from typing import Union, List

from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import download_and_extract_archive

from pytorch_lightning import LightningDataModule

__all__ = ['ImagenetteDataModule']

# DATADIR = Path('data/imagewoof2/')
DATADIR = Path('data/')

imagenette_urls = {'imagenette2': 'https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz',
                   'imagewoof2': 'https://s3.amazonaws.com/fast-ai-imageclas/imagewoof2.tgz'}

imagenette_len = {'imagenette2': {'train': 9469, 'val': 3925},
                  'imagewoof2': {'train': 9025, 'val': 3929}
                  }

imagenette_md5 = {'imagenette2': '43b0d8047b7501984c47ae3c08110b62',
                  'imagewoof2': '5eaf5bbf4bf16a77c616dc6e8dd5f8e9'}

normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

def check_data_exists(root, name) -> bool:
    ''' Verify data at root and return True if len of images is Ok.
    '''
    num_classes = 10
    if not root.exists():
        return False

    for split in ['train', 'val']:
        split_path = Path(root, split)
        if not root.exists():
            return False

        classes_dirs = [dir_entry for dir_entry in os.scandir(split_path)
                        if dir_entry.is_dir()]
        if num_classes != len(classes_dirs):
            return False

        num_samples = 0
        for dir_entry in classes_dirs:
            num_samples += len([fn for fn in os.scandir(dir_entry)
                                if fn.is_file()])

        if num_samples != imagenette_len[name][split]:
            return False

    return True


def train_transform(image_size, train_img_scale=(0.35, 1)):
    """
    The standard imagenet transforms: random crop, resize to self.image_size, flip.
    """
    preprocessing = T.Compose([
        T.RandomResizedCrop(image_size, scale=train_img_scale),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalize,
    ])

    return preprocessing

def val_transform(image_size):
    """
    The standard imagenet transforms for validation: central crop, resize to self.image_size.
    """
    preprocessing = T.Compose([
        T.Resize(image_size + 32),
        T.CenterCrop(image_size),
        T.ToTensor(),
        normalize,
    ])
    return preprocessing


class ImagenetteDataModule(LightningDataModule):
    '''Imagenette dataset Datamodule.
    Subset of ImageNet.
    https://github.com/fastai/imagenette
    '''

    def __init__(
            self,
            data_dir: str = DATADIR,
            image_size: int = 192,
            num_workers: int = 4,
            batch_size: int = 32,
            woof: bool = False,
            train_transforms: Union[None, List] = None,
            val_transforms: Union[None, List] = None,
            ):
        '''
        Args:
            data_dir: path to datafolder
        '''
        super().__init__()
        self.image_size = image_size
        self.dims = (3, self.image_size, self.image_size)
        self.data_dir = Path(data_dir)
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.num_classes = 10
        self.train_img_scale = (0.35, 1)
        self.woof = woof
        self.name = 'imagewoof2' if woof else 'imagenette2'
        self.root = Path(self.data_dir, self.name)
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms


    def prepare_data(self):
        """ Download data if no data at root
        """
        if not check_data_exists(self.root, self.name):
            dataset_url = imagenette_urls[self.name]
            download_and_extract_archive(url=dataset_url, download_root=self.data_dir, md5=imagenette_md5[self.name])

    def setup(self, stage=None):
        train_transforms = train_transform(self.image_size, self.train_img_scale) if self.train_transforms is None else self.train_transforms
        self.train_dataset = ImageFolder(root=Path(self.root, 'train'), transform=train_transforms)
        val_transforms = val_transform(self.image_size) if self.val_transforms is None else self.val_transforms
        self.val_dataset = ImageFolder(root=Path(self.root, 'val'), transform=val_transforms)

    def train_dataloader(self):
        """
        Uses the train split of dataset
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
        Uses the valid part of the dataset
        """

        loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        return loader


class ImageWoofDataModule(ImagenetteDataModule):
    '''ImageWoof dataset Datamodule,
    Part of Imagenette dataset.
    Subset of ImageNet.
    https://github.com/fastai/imagenette
    '''

    def __init__(
            self,
            data_dir: str = DATADIR,
            image_size: int = 192,
            num_workers: int = 4,
            batch_size: int = 32,
            train_transforms: Union[None, List] = None,
            val_transforms: Union[None, List] = None,):
        '''
        Args:
            data_dir: path to datafolder
        '''
        super().__init__(woof=True,
                         data_dir=data_dir,
                         image_size=image_size,
                         num_workers=num_workers,
                         batch_size=batch_size,
                         train_transforms=train_transforms,
                         val_transforms=val_transforms)
