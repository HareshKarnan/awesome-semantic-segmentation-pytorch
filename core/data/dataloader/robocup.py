"""YCB Object Semantic Segmentation Dataset."""
import os
import torch
import numpy as np
import glob
from PIL import Image
from .segbase import SegmentationDataset


class RobocupSegmentation(SegmentationDataset):
    """YCB Semantic Segmentation Dataset.

    Parameters
    ----------
    root : string
        Path to YCB folder. Default is './datasets/ycb'
    split: string
        'train', 'val' or 'test'
    transform : callable, optional
        A function that transforms the image
    Examples
    --------
    >>> from torchvision import transforms
    >>> import torch.utils.data as data
    >>> # Transforms for Normalization
    >>> input_transform = transforms.Compose([
    >>>     transforms.ToTensor(),
    >>>     transforms.Normalize((.485, .456, .406), (.229, .224, .225)),
    >>> ])
    >>> # Create Dataset
    >>> trainset = ADE20KSegmentation(split='train', transform=input_transform)
    >>> # Create Training Loader
    >>> train_data = data.DataLoader(
    >>>     trainset, 4, shuffle=True,
    >>>     num_workers=4)
    """
    BASE_DIR = 'YCBSubData'
    NUM_CLASS = 8

    def __init__(self, root='../datasets/robocup_subset', split='test', mode=None, transform=None, **kwargs):
        super(RobocupSegmentation, self).__init__(root, split, mode, transform, **kwargs)
        # root = os.path.join(root, self.BASE_DIR)
        self.root = root
        assert os.path.exists(root), "Please download the robocup_subset training data into the folder ../datasets/robocup_subset"
        self.images, self.masks = _get_image_pairs(root, split)
        assert (len(self.images) == len(self.masks))
        if len(self.images) == 0:
            raise RuntimeError("Found 0 images in subfolders of:" + root + "\n")
        print('Found {} images in the folder {}'.format(len(self.images), root))

        self.valid_classes = list(np.linspace(-1, 150, 9).astype(np.int32))[1:]
        self._key = np.arange(1, len(self.valid_classes)+1)
        self.mapping = {self.valid_classes[i]:self._key[i] for i in range(len(self.valid_classes))}

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        if self.mode == 'test':
            img = self._img_transform(img)
            if self.transform is not None:
                img = self.transform(img)
            return img, os.path.basename(self.images[index])
        mask = Image.open(self.masks[index])
        # synchrosized transform
        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        elif self.mode == 'val':
            img, mask = self._val_sync_transform(img, mask)
        else:
            assert self.mode == 'testval'
            img, mask = self._img_transform(img), self._mask_transform(mask)
        # general resize, normalize and to Tensor
        if self.transform is not None:
            img = self.transform(img)
        return img, mask, os.path.basename(self.images[index])

    def _classes_2_index(self, mask):
        for class_num in self.valid_classes:
            mask = np.where(mask==class_num, self.mapping[class_num], mask)
        return mask

    def _mask_transform(self, mask):
        return torch.LongTensor(self._classes_2_index(mask).astype('int32'))

    def __len__(self):
        return len(self.images)

    @property
    def pred_offset(self):
        return 1

    def classes(self):
        """Category names."""

        return ("chips_can", "gelatin_box", "potted_meat_can", "pudding_box", "red_cup", "soylent", "tomato_soup_can", \
                "tuna_fish_can")


def _get_image_pairs(folder, mode='train'):
    img_paths = []
    mask_paths = []
    if mode == 'train':
        img_folder = os.path.join(folder, 'train/rgb')
    else:
        img_folder = os.path.join(folder, 'test/rgb')

    img_paths = glob.glob(img_folder+'/*.jpg', recursive=True)

    for img_path in img_paths :
        mask_path = img_path.replace('rgb', 'mask')
        mask_path = ".".join(mask_path.split('.')[:-1])+'.png'
        mask_paths.append(mask_path)

    assert len(img_paths)==len(mask_paths), "Dataset Corrupted. Please make the dataset again using process_dataset.py"

    return img_paths, mask_paths


if __name__ == '__main__':
    train_dataset = RobocupSegmentation()
