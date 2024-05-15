import os
import cv2
import time
import torch
from enum import Enum
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from config import AutoEncoderKL60_Config, VanillaVAE_Config, ImprovedVAE_Config, ImprovedVAE_Imagenet_Config
from model.AutoEncoderKL import autoencoder_kl60
from model.VanillaVAE import vanilla_vae
from model.ImprovedVAE import improved_vae
from dataloader import ImagenetDataset, CelebADataset, CenterCropResizer, RandSampler, FixCollector, RandomCropResizer

def reconstruct(model, config, eval_dataset, ptdtype, device, sample_posterior):
    eval_sampler = RandSampler(eval_dataset, batch_size=config.batch_size, drop_last=True, shuffle=True)
    eval_loader = DataLoader(eval_dataset, num_workers=4, pin_memory=True, collate_fn=FixCollector, batch_sampler=eval_sampler)

    with torch.no_grad():
        for i, data in enumerate(eval_loader, 0):
            input = data.to(device, non_blocking=True)
            with autocast(dtype=ptdtype):
                generates = model(input, sample_posterior=sample_posterior)
                generates = (generates + 1) * 127.5
                generates = generates.permute(0, 2, 3, 1).cpu().to(dtype=torch.uint8).numpy()
                input = (input + 1) * 127.5
                input = input.permute(0, 2, 3, 1).cpu().to(dtype=torch.uint8).numpy()

                for j in range(config.batch_size):
                    img = generates[j]
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                    ori_img = input[j]
                    ori_img = cv2.cvtColor(ori_img, cv2.COLOR_RGB2BGR)
                    
                    cv2.imshow('x', ori_img)
                    cv2.imshow('y', img)
                    cv2.waitKey(0)

def crop_source(images, target, image_num=5000, transform=None):
    for name in images[:image_num]:
        img = cv2.imread(name)
        if transform:
            img = transform(img) 
        name = name.split('/')[-1]
        if name[-4:] == "JPEG":
            name = name[:-4] + "jpg"
        cv2.imwrite(os.path.join(target, name), img)

def generate(model, config, image_dir, image_num, batch_size, ptdtype, device):
    with torch.no_grad():
        for i in range(image_num // batch_size):
            with autocast(dtype=ptdtype):
                inputs = torch.randn([batch_size, config.z_channel, config.img_size//16, config.img_size//16]).to(device, non_blocking=True)
                if hasattr(model, 'decode'):
                    generates = model.decode(inputs)
                else:
                    generates = model.module.decode(inputs)
                generates = (generates + 1) * 127.5
                generates = generates.permute(0, 2, 3, 1).cpu().to(dtype=torch.uint8).numpy()
                for j in range(batch_size):
                    img = generates[j]
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    file_name = f"{i * batch_size + j}.jpg"
                    file_name = os.path.join(image_dir, file_name)
                    cv2.imwrite(file_name, img)

def cal_fid(source, target, model, config, ptdtype, device, image_num=5000, batch_size=64):
    generate(model, config, target, image_num, batch_size, ptdtype, device)
    os.system(f"python -m pytorch_fid {source} {target}")

if __name__ == "__main__":
    config = ImprovedVAE_Imagenet_Config()
    # config = ImprovedVAE_Config()
    arch = improved_vae
    image_num = 5000
    batch_size = 64
    gpu = 0
    dtype = "bfloat16"
    # data_path = "/home/work/imagenet"
    data_path = "/home/work/CelebA"
    cur_dir = "/home/work/disk/vision/autoencoder"
    model_path = "checkpoint/improved_vae_imagenet256_gan_pl_loss0.63.pth.tar"
    # model_path = "checkpoint/improved_vae_celebA176_gan_pl_loss_fid32.pth.tar"
    model_path = os.path.join(cur_dir, model_path)
    image_dir = os.path.join(cur_dir, 'image')

    torch.cuda.set_device(gpu)
    device = torch.device(f'cuda:{gpu}')
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    model = arch(config, True, model_path)
    model.cuda(gpu)
    model.eval()

    # eval_dataset = ImagenetDataset(data_path, 'val', transform=RandomCropResizer(config.img_size))
    # eval_dataset = ImagenetDataset(data_path, 'val', transform=CenterCropResizer(config.img_size, config.crop_size))
    eval_dataset = CelebADataset(data_path, 'val', transform=CenterCropResizer(config.img_size, config.crop_size))

    # generate(model, config, image_dir, image_num, batch_size, ptdtype, device)
    reconstruct(model, config, eval_dataset, ptdtype, device, sample_posterior=False)
    # crop_source(eval_dataset.images, os.path.join(data_path, 'crop'), transform=CenterCropResizer(config.img_size, config.crop_size))
    # cal_fid(os.path.join(data_path, 'crop'), os.path.join(cur_dir, 'image'), model, config, ptdtype, device)