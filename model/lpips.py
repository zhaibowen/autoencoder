import os
import sys
import torch
import torch.nn as nn
import numpy as np
from .ResNet import resnet18
from torch.utils.data import DataLoader
import torch.nn.functional as functional

class ScalingLayer(nn.Module):
    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.register_buffer('shift', torch.Tensor([0.485, 0.456, 0.406])[None, :, None, None])
        self.register_buffer('scale', torch.Tensor([0.229, 0.224, 0.225])[None, :, None, None])
        self.register_buffer('shift2', torch.Tensor([1.0, 1.0, 1.0])[None, :, None, None])
        self.register_buffer('scale2', torch.Tensor([2.0, 2.0, 2.0])[None, :, None, None])

    def forward(self, x):
        x = (x + self.shift2) / self.scale2
        x = (x - self.shift) / self.scale
        return x

def normalize_tensor(x, eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(x**2, dim=1, keepdim=True))
    return x / (norm_factor + eps)

def spatial_average(x, keepdim=True):
    return x.mean([1, 2, 3], keepdim=keepdim)

class LPIPS(nn.Module):
    def __init__(self, model_path, test=False):
        super().__init__()
        self.test = test
        self.scaling_layer = ScalingLayer()
        self.net = resnet18(pretrained=True, model_path=model_path, test=test)
        self.net.eval()
        for param in self.net.parameters():
            param.requires_grad = False

    def get_num_params(self):
        n_params = sum(p.numel() for p in self.parameters())
        return n_params
    
    def forward(self, x, y=None):
        x = self.net(self.scaling_layer(x))
        if self.test:
            return x

        y = self.net(self.scaling_layer(y))
        res = []
        for i1, i2 in zip(x, y):
            f1, f2 = normalize_tensor(i1), normalize_tensor(i2)
            diff = (f1 - f2) ** 2
            avg = spatial_average(diff, keepdim=True)
            res.append(avg)
        res = torch.cat(res, dim=1).sum()
        return res

def get_index_label(index_path, name_path):
    tag2index = {}
    index2label = {}

    with open(index_path, 'r') as f:
        x = f.readlines()
        x = list(map(lambda x: x.strip(), x))
        for index, tag in enumerate(x):
            tag2index[tag] = index

    with open(name_path, 'r') as f:
        x = f.readlines()
        x = list(map(lambda x: x.strip().split(' ', 2), x))
        for i, tag, label in x:
            index = tag2index[tag]
            index2label[index] = label

    return index2label

def test_resnet():
    data_path = '/home/work/imagenet'
    eval_dataset = CelebADataset(data_path, 'val/n01829413', transform=CenterCropResizer(256, 256))
    eval_sampler = RandSampler(eval_dataset, batch_size=10, drop_last=True, shuffle=False)
    eval_loader = DataLoader(eval_dataset, num_workers=4, pin_memory=True, collate_fn=FixCollector, batch_sampler=eval_sampler)

    model = LPIPS(model_path='/home/work/disk/vision/classification/checkpoint/resnet18_acc69.3_s.pth.tar', test=True)
    index2label = get_index_label(index_path='/home/work/disk/vision/classification/imagenet.index', name_path='/home/work/disk/vision/classification/imagenet.names')

    for i, data in enumerate(eval_loader, 0):
        output = model(data)
        output = functional.softmax(output, 1)
        output = output.numpy()
        index = np.argmax(output, 1)
        for j in range(index.shape[0]):
            print(output[j, index[j]], index2label[index[j]])
        break

if __name__ == "__main__":
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from dataloader import CelebADataset, CenterCropResizer, RandSampler, FixCollector

    test_resnet()