import random
import os
import numpy as np

from PIL import Image
from torch.utils import data
from torchvision import transforms as T
from torchvision.transforms import functional as F

from data_aug import data_aug

# from loader.img_mask_aug import data_aug
# from tnscui_utils.TNSUCI_util import char_color

sep = os.sep


def get_data_loader(
    data_dir: str, batch_size=32, shuffle=True, num_workers=4, preprocess_fn=None, pin_mem=False
) -> tuple[data.DataLoader, data.DataLoader, data.DataLoader]:
    """
    Get data loaders for training, validation, and testing.
    Args:
        data_dir (str): Directory containing the dataset.
        batch_size (int): Batch size for data loaders.
        shuffle (bool): Whether to shuffle the dataset.
        num_workers (int): Number of worker threads for data loading.
        preprocess_fn (callable): Preprocessing function to apply to the images.
        pin_mem (bool): Whether to pin memory for data loaders.
    Returns:
        tuple: Data loaders for training, validation, and testing.
    """
    train_set = DataLoader(data_dir, "train", preprocess_fn=preprocess_fn)
    val_set = DataLoader(data_dir, "val", preprocess_fn=preprocess_fn, with_aug=False)
    test_set = DataLoader(data_dir, "test", preprocess_fn=preprocess_fn, with_aug=False)

    train_loader = data.DataLoader(
        train_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_mem
    )
    val_loader = data.DataLoader(
        val_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_mem
    )
    test_loader = data.DataLoader(
        test_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_mem
    )

    return train_loader, val_loader, test_loader

# 224, 224
class DataLoader(data.Dataset):
    def __init__(self, data_dir, split, crop_size=(1568, 1568), preprocess_fn=None, with_aug=True):
        self.with_aug = with_aug
        self.data_dir = data_dir
        self.crop_size = crop_size
        self.scale_size = (224, 224)
        x_path = os.path.join(data_dir, split)
        y_path = os.path.join(data_dir, split + "annot")
        self.images = [os.path.join(x_path, i) for i in os.listdir(x_path)]
        self.labels = [os.path.join(y_path, i) for i in os.listdir(y_path)]
        assert len(self.images) == len(
            self.labels
        ), "Mismatch between images and labels"
        self.preprocess_fn = preprocess_fn

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = np.array(self._im_crop(Image.open(self.images[idx]), "L"))
        label = np.array(self._im_crop(Image.open(self.labels[idx]), "1"))
        if self.with_aug:
            image, label = data_aug(image, label)
        image = self.preprocess_fn(image) if self.preprocess_fn else image
        return np.expand_dims(image, axis=0), np.expand_dims(label, axis=0)

    def _im_crop(self, im: Image.Image, mode) -> Image.Image:
        """
        Crop the image to the same size(Center).
        """
        # im = im.convert(mode)
        # im.resize(self.crop_size, Image.BILINEAR)
        w, h = im.size
        crop_w, crop_h = self.crop_size
        x1 = int((w - crop_w) / 2)
        # y1 = int((h - crop_h) / 2)
        y1 = 0
        x2 = int((w + crop_w) / 2)
        # y2 = int((h + crop_h) / 2)
        y2 = y1 + crop_h
        # return im.crop((x1, y1, x2, y2))
        # return im.crop((x1, y1, x2, y2)).resize(self.scale_size).convert("L")
        return im.crop((x1, y1, x2, y2)).resize(self.scale_size).convert(mode)
        
