import os
import cv2
import torch
import random
import numpy as np
import albumentations
import torch.distributed as dist
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler

class ImagenetDataset(Dataset):
    def __init__(self, root_dir, set_name, transform=None):
        self.transform = transform
        self.images = []

        img_dirs = os.listdir(os.path.join(root_dir, set_name))
        for img_dir in img_dirs:
            imgs = os.listdir(os.path.join(root_dir, set_name, img_dir))
            imgs = list(map(lambda x: os.path.join(root_dir, set_name, img_dir, x), imgs))
            self.images.extend(imgs)

    def __getitem__(self, index):
        img = cv2.imread(self.images[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(img)
        img = torch.from_numpy(img).permute(2, 0, 1).contiguous()
        img = img / 127.5 - 1.0
        return img

    def __len__(self):
        return len(self.images)

class CelebADataset(Dataset):
    def __init__(self, root_dir, set_name, transform=None):
        self.transform = transform
        self.images = []

        imgs = os.listdir(os.path.join(root_dir, set_name))
        imgs = list(map(lambda x: os.path.join(root_dir, set_name, x), imgs))
        self.images = imgs

    def __getitem__(self, index):
        img = cv2.imread(self.images[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(img)
        img = torch.from_numpy(img).permute(2, 0, 1).contiguous()
        img = img / 127.5 - 1.0
        return img

    def __len__(self):
        return len(self.images)

class RandomCropResizer(object):
    def __init__(self, img_size, min_crop_f=0.5, max_crop_f=1.0):
        self.img_size = img_size
        self.min_crop_f = min_crop_f
        self.max_crop_f = max_crop_f
        self.image_rescaler = albumentations.SmallestMaxSize(max_size=img_size, interpolation=cv2.INTER_AREA)
        self.hflip = albumentations.HorizontalFlip()

    def __call__(self, image):
        H, W, _ = image.shape
        # cv2.imshow('x', image)
        image = self.hflip(image=image)["image"]
        min_side_len = min(H, W)
        crop_side_len = min_side_len * np.random.uniform(self.min_crop_f, self.max_crop_f, size=None)
        crop_side_len = int(crop_side_len)
        cropper = albumentations.RandomCrop(height=crop_side_len, width=crop_side_len)
        image = cropper(image=image)["image"]
        image = self.image_rescaler(image=image)["image"]
        # print(H, W, crop_side_len)
        # cv2.imshow('y', image)
        # cv2.waitKey(0)
        return image

class CenterCropResizer(object):
    def __init__(self, img_size, crop_size=None):
        self.img_size = img_size
        self.crop_size = crop_size
        self.image_rescaler = albumentations.SmallestMaxSize(max_size=img_size, interpolation=cv2.INTER_AREA)
        self.hflip = albumentations.HorizontalFlip()

    def __call__(self, image):
        H, W, _ = image.shape
        # cv2.imshow('x', image)
        image = self.hflip(image=image)["image"]
        crop_size = min(H, W) if self.crop_size is None else min(min(H, W), self.crop_size)
        cropper = albumentations.CenterCrop(height=crop_size, width=crop_size)
        image = cropper(image=image)["image"]
        image = self.image_rescaler(image=image)["image"]
        # print(H, W)
        # cv2.imshow('y', image)
        # cv2.waitKey(0)
        return image

class RandSampler(Sampler):
    def __init__(self, data_source, batch_size, drop_last, shuffle=True):
        self.batch_size = batch_size
        self.order = list(range(len(data_source)))
        self.total_size = len(self.order) - len(self.order) % self.batch_size
        if not drop_last: self.total_size += batch_size

        if shuffle: random.shuffle(self.order)
        self.groups = []
        for i in range(0, self.total_size, self.batch_size):
            self.groups.append([self.order[x % len(self.order)] for x in range(i, i + self.batch_size)])

    def shuffle(self, epoch=0):
        random.shuffle(self.order)
        self.groups = []
        for i in range(0, self.total_size, self.batch_size):
            self.groups.append([self.order[x % len(self.order)] for x in range(i, i + self.batch_size)])

    def __iter__(self):
        for group in self.groups:
            yield group

    def __len__(self):
        return len(self.groups)

class DistRandSampler(Sampler):
    def __init__(self, data_source, batch_size, drop_last, shuffle=True):
        self.rank = dist.get_rank()
        self.num_replicas = dist.get_world_size()
        self.batch_size = batch_size
        self.order = list(range(len(data_source)))
        self.total_size = len(self.order) - len(self.order) % (self.num_replicas * self.batch_size)
        if not drop_last: self.total_size += self.num_replicas * self.batch_size

        if shuffle:
            g = torch.Generator()
            g.manual_seed(-1)
            self.order = torch.randperm(len(self.order), generator=g).tolist()
        self.groups = []
        for i in range(self.rank * self.batch_size, self.total_size, self.num_replicas * self.batch_size):
            self.groups.append([self.order[x % len(self.order)] for x in range(i, i + self.batch_size)])

    def shuffle(self, epoch):
        g = torch.Generator()
        g.manual_seed(epoch)
        self.order = torch.randperm(len(self.order), generator=g).tolist()
        self.groups = []
        for i in range(self.rank * self.batch_size, self.total_size, self.num_replicas * self.batch_size):
            self.groups.append([self.order[x % len(self.order)] for x in range(i, i + self.batch_size)])

    def __iter__(self):
        for group in self.groups:
            yield group

    def __len__(self):
        return len(self.groups)

def FixCollector(batch):
    images = torch.stack(batch)
    return images

if __name__ == "__main__":
    root_dir = "/home/work/CelebA"
    # dataset = ImagenetDataset(root_dir, 'train', transform=CenterCropResizer(256))
    dataset = CelebADataset(root_dir, transform=CenterCropResizer(64, crop_size=148))
    sampler = RandSampler(dataset, batch_size=4, drop_last=True, shuffle=True)
    dataloader = DataLoader(dataset, num_workers=0, pin_memory=True, collate_fn=FixCollector, batch_sampler=sampler)

    for i, data in enumerate(dataloader):
        a = 1