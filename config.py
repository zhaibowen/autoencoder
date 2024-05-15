from dataclasses import dataclass

@dataclass
class AutoEncoderKL60_Config:
    gpu_num = 3
    batch_size = 21
    gradient_accumulation_steps = 1
    num_epoch = 30
    learning_rate = 6e-4
    min_lr = 6e-5
    beta1 = 0.9
    beta2 = 0.95
    kl_weight = 0.01
    rec_weight = 0.01
    weight_decay = 1e-1
    warmup_iters = 10
    max_iters = 6010
    lr_decay_iters = 6000
    grad_clip = 1.0

    layers = [2, 2, 2, 2]
    # channels = [128, 128, 256, 512, 512] # 84M
    channels = [64, 64, 128, 256, 256] # 21M
    z_channel = 4
    embed_dim = 4
    img_size = 256
    compress_ratio = 8

@dataclass
class ImprovedVAE_Config:
    gpu_num = 3
    batch_size = 32
    gradient_accumulation_steps = 1
    num_epoch = 60
    learning_rate = 6e-4
    min_lr = 1e-5
    beta1 = 0.9
    beta2 = 0.95
    warmup_iters = 1000
    max_iters = 112000
    lr_decay_iters = 112000
    disc_start = 10000
    grad_clip = 1.0

    layers = [3, 3, 3, 3]
    channels = [64, 64, 128, 256, 256]
    z_channel = 8
    img_size = 176
    crop_size = 176
    kl_weight = 0.0005
    pl_weight = 0.1
    disc_weight = 0.5

@dataclass
class ImprovedVAE_Imagenet_Config:
    gpu_num = 3
    batch_size = 32
    gradient_accumulation_steps = 2
    num_epoch = 60
    learning_rate = 6e-4
    min_lr = 1e-5
    beta1 = 0.9
    beta2 = 0.95
    warmup_iters = 1000
    max_iters = 400000
    lr_decay_iters = 400000
    disc_start = 40000
    grad_clip = 1.0

    layers = [3, 3, 3, 3]
    channels = [64, 64, 128, 256, 256]
    z_channel = 16
    img_size = 256
    crop_size = 256
    kl_weight = 0.000005
    pl_weight = 0.1
    disc_weight = 0.5

@dataclass
class VanillaVAE_Config:
    gpu_num = 3
    batch_size = 32
    gradient_accumulation_steps = 1
    num_epoch = 100
    learning_rate = 6e-4
    min_lr = 6e-5
    beta1 = 0.9
    beta2 = 0.95
    weight_decay = 0.0
    warmup_iters = 10
    max_iters = 210000
    lr_decay_iters = 200000
    grad_clip = 1.0

    channels = [32, 64, 128, 256, 512]
    latent_dim = 128
    img_size = 64
    crop_size = 148
    kl_weight = 0.00025