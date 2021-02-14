import os
import torch
import numpy as np
import torchvision.transforms as trns
from PIL import Image
from scipy.io import loadmat
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from config import path_config
import matplotlib.pyplot as plt


class Synth90kDataset(Dataset):
    CHARS = '0123456789abcdefghijklmnopqrstuvwxyz'
    CHAR2LABEL = {char: i + 1 for i, char in enumerate(CHARS)}
    LABEL2CHAR = {label: char for char, label in CHAR2LABEL.items()}

    def __init__(self, root_dir=None, mode=None, paths=None, img_height=32, img_width=100):
        if root_dir and mode and not paths:
            paths, texts = self._load_from_raw_files(root_dir, mode)
        elif not root_dir and not mode and paths:
            texts = None

        self.paths = paths
        self.texts = texts
        self.img_height = img_height
        self.img_width = img_width

    def _load_from_raw_files(self, root_dir, mode):
        mapping = {}
        with open(os.path.join(root_dir, 'lexicon.txt'), 'r') as fr:
            for i, line in enumerate(fr.readlines()):
                mapping[i] = line.strip()

        paths_file = None
        if mode == 'train':
            paths_file = 'annotation_train.txt'
        elif mode == 'dev':
            paths_file = 'annotation_val.txt'
        elif mode == 'test':
            paths_file = 'annotation_test.txt'

        paths = []
        texts = []
        with open(os.path.join(root_dir, paths_file), 'r') as fr:
            for line in fr.readlines():
                path, index_str = line.strip().split(' ')
                path = os.path.join(root_dir, path)
                index = int(index_str)
                text = mapping[index]
                paths.append(path)
                texts.append(text)
        return paths, texts

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]

        try:
            image = Image.open(path).convert('L')  # grey-scale
        except IOError:
            print('Corrupted image for %d' % index)
            return self[index + 1]

        image = image.resize((self.img_width, self.img_height), resample=Image.BILINEAR)
        image = np.array(image)
        image = image.reshape((1, self.img_height, self.img_width))
        image = (image / 127.5) - 1.0

        image = torch.FloatTensor(image)
        if self.texts:
            text = self.texts[index]
            target = [self.CHAR2LABEL[c] for c in text]
            target_length = [len(target)]

            target = torch.LongTensor(target)
            target_length = torch.LongTensor(target_length)
            return image, target, target_length
        else:
            return image


def synth90k_collate_fn(batch):
    images, targets, target_lengths = zip(*batch)
    images = torch.stack(images, 0)
    targets = torch.cat(targets, 0)
    target_lengths = torch.cat(target_lengths, 0)
    return images, targets, target_lengths


def process_associate_txt(dataset_dir, associate_txt):
    imgs, depths, labls = [], [], []
    for x in associate_txt:
        row = x.split(" ")
        imgs.append(os.path.join(dataset_dir, row[1]))
        depths.append(os.path.join(dataset_dir, row[3]))
        labls.append([float(x) for x in row[5:]])
    return imgs, depths, labls


# Create train/valid transforms
train_transform = trns.Compose([
    trns.ToTensor(),
    # trns.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
normalize_transform = trns.Compose([
    trns.ToTensor(),
    trns.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class TUM_Dataset(Dataset):
    def __init__(self, split="train", transform=train_transform):
        """
        :param split: train, dev
        :param transform: Transform preprocessing
        """
        self.transform = transform
        self.dataset_dir = path_config[f'{split}_dataset_dir']

        # Load image path and annotations
        associate_txt_path = os.path.join(self.dataset_dir, 'associate_all.txt')
        with open(associate_txt_path, 'r') as f:
            images, depths, labels = process_associate_txt(self.dataset_dir, f.readlines())

        self.image_paths = images
        self.depth_paths = depths
        self.lbls = torch.tensor(labels, dtype=torch.float32)
        assert len(self.image_paths) == len(self.lbls), 'mismatched dataset length!'
        print('Total data in {} split: {}'.format(split, len(self.image_paths)))

    def __getitem__(self, index):
        # --------------------------------------------
        # 1. Read from file (using numpy.fromfile, PIL.Image.open)
        # 2. Preprocess the data (torchvision.Transform)
        # 3. Return the data (e.g. image and label)
        # --------------------------------------------
        label = self.lbls[index]

        imgpath_1 = self.image_paths[index]
        img_1 = Image.open(imgpath_1).convert('RGB')

        if index == 0:
            img_0 = img_1
        else:
            imgpath_0 = self.image_paths[index-1]
            img_0 = Image.open(imgpath_0).convert('RGB')

        return self.transform(img_0), self.transform(img_1), label

    def __len__(self):
        # --------------------------------------------
        # Indicate the total size of the dataset
        # --------------------------------------------
        return len(self.image_paths)


if __name__ == "__main__":
    # Create train/valid datasets
    train_set = TUM_Dataset(split='train', transform=train_transform)
    valid_set = TUM_Dataset(split='eval', transform=train_transform)

    # Create train/valid loaders
    train_loader = DataLoader(
        dataset=train_set, batch_size=16, shuffle=False, num_workers=4)
    valid_loader = DataLoader(
        dataset=valid_set, batch_size=16, shuffle=False, num_workers=4)

    # Get images and labels in a mini-batch of train_loader
    for imgs, lbls in train_loader:
        # plt.imshow(imgs[0].permute(1,2,0))
        # plt.show()
        print('Size of image:', imgs.size())  # batch_size * 3 * 224 * 224
        # print('Type of image:', imgs.dtype)  # float32
        print('Size of label:', lbls.size())  # batch_size
        print('Type of label:', lbls.dtype)  # int64(long)
        break
