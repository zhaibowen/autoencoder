import os
import cv2
import time
import math
import torch
import inspect
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from config import AutoEncoderKL60_Config, VanillaVAE_Config, ImprovedVAE_Config, ImprovedVAE_Imagenet_Config
from model.AutoEncoderKL import autoencoder_kl60
from model.VanillaVAE import vanilla_vae
from model.ImprovedVAE import improved_vae
from dataloader import ImagenetDataset, CelebADataset, RandomCropResizer, CenterCropResizer, RandSampler, DistRandSampler, FixCollector
from infer import cal_fid, crop_source
torch.backends.cudnn.benchmark = True

def get_lr(it, config):
    if it < config.warmup_iters:
        return config.learning_rate * it / config.warmup_iters
    if it > config.lr_decay_iters:
        return config.min_lr
    decay_ratio = (it - config.warmup_iters) / (config.lr_decay_iters - config.warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)

def train(train_loader, model, device, optimizer, scaler, ptdtype, is_master, config, iter_num):
    rec_loss_knt = 0
    kl_loss_knt = 0
    pl_loss_knt = 0
    loss_knt = 0
    g_loss_knt = 0
    d_loss_knt = 0
    d_weight_knt = 0
    loss_num_g = 0
    loss_num_d = 0

    model.train()
    st = time.time()
    for opt in optimizer:
        opt.zero_grad(set_to_none=True)
    for i, data in enumerate(train_loader):
        lr = get_lr(iter_num, config)
        need_g_loss = iter_num > config.disc_start
        optimizer_idx = iter_num % 2 if need_g_loss else 0
        opt = optimizer[optimizer_idx]
        for param_group in opt.param_groups:
            param_group['lr'] = lr
        input = data.to(device, non_blocking=True)
        with autocast(dtype=ptdtype):
            rec_loss, kl_loss, pl_loss, loss, d_weight, g_loss, d_loss = model(input, optimizer_idx=optimizer_idx, need_g_loss=need_g_loss)
        
        if optimizer_idx == 0:
            loss /= config.gradient_accumulation_steps
            scaler.scale(loss).backward()
            loss_knt += loss.item()
            rec_loss_knt += rec_loss.item() / config.gradient_accumulation_steps
            kl_loss_knt += kl_loss.item() / config.gradient_accumulation_steps
            pl_loss_knt += pl_loss.item() / config.gradient_accumulation_steps
            g_loss_knt += g_loss.item() / config.gradient_accumulation_steps
            d_weight_knt += d_weight / config.gradient_accumulation_steps
        else:
            d_loss /= config.gradient_accumulation_steps
            scaler.scale(d_loss).backward()
            d_loss_knt += d_loss.item()

        if (i + 1) % config.gradient_accumulation_steps == 0:
            if optimizer_idx == 0:
                loss_num_g += 1
            else:
                loss_num_d += 1
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            iter_num += 1
            if iter_num % 500 == 0:
                if is_master:
                    print(f"step {iter_num}, "
                          f"loss {loss_knt/loss_num_g:.3f}, "
                          f"rec {rec_loss_knt/loss_num_g:.3f}, "
                          f"kl {kl_loss_knt/loss_num_g:.3f}, "
                          f"pl {pl_loss_knt/loss_num_g:.3f}, "
                          f"g {g_loss_knt/loss_num_g:.3f}, "
                          f"w {d_weight_knt/loss_num_g:.3f}, "
                          f"d {d_loss_knt/loss_num_d if loss_num_d else 0:.3f}, "
                          f"lr: {lr:.7f}, "
                          f"consume {time.time()-st:.2f}s")
                st = time.time()
                rec_loss_knt = 0
                kl_loss_knt = 0
                pl_loss_knt = 0
                loss_knt = 0
                g_loss_knt = 0
                d_weight_knt = 0
                d_loss_knt = 0
                loss_num_g = 0
                loss_num_d = 0
            if iter_num >= config.max_iters:
                break
    return iter_num

def validate(valid_loader, model, device, ptdtype):
    rec_loss_knt = 0
    kl_loss_knt = 0
    pl_loss_knt = 0
    running_loss = 0
    g_loss_knt = 0
    d_loss_knt = 0
    d_weight_knt = 0
    count = 0

    model.eval()
    st = time.time()
    with torch.no_grad():
        for i, data in enumerate(valid_loader, 0):
            input = data.to(device, non_blocking=True)
            with autocast(dtype=ptdtype):
                rec_loss, kl_loss, pl_loss, loss, d_weight, g_loss, d_loss = model(input, optimizer_idx=2, need_g_loss=True)
                running_loss += loss.item()
                rec_loss_knt += rec_loss.item()
                kl_loss_knt += kl_loss.item()
                pl_loss_knt += pl_loss.item()
                g_loss_knt += g_loss.item()
                d_loss_knt += d_loss.item()
                d_weight_knt += d_weight
                count += 1

    print(f"    valid loss: {running_loss / count:.3f}, "
          f"rec {rec_loss_knt/count:.3f}, "
          f"kl {kl_loss_knt/count:.3f}, "
          f"pl {pl_loss_knt/count:.3f}, "
          f"g {g_loss_knt/count:.3f}, "
          f"w {d_weight_knt/count:.3f}, "
          f"d {d_loss_knt/count:.3f}, "
          f"consume: {time.time() - st:.3f}s")
    return running_loss / count

def main(gpu, gpu_num, config, distributed, evaluate, load_model, save_model, arch, dtype, data_path, cur_dir, model_path, pl_model_path, trained_epoch, compile):
    model_path = os.path.join(cur_dir, model_path)
    is_master = distributed == False or gpu == 0

    if distributed:
        dist.init_process_group(backend='nccl', init_method='tcp://localhost:23456', world_size=gpu_num, rank=gpu)

    torch.cuda.set_device(gpu)
    device = torch.device(f'cuda:{gpu}')
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    model = arch(config, load_model, model_path, pl_model_path)
    model.cuda(gpu)

    eval_dataset = ImagenetDataset(data_path, 'val', transform=CenterCropResizer(config.img_size, config.crop_size))
    # eval_dataset = CelebADataset(data_path, "val", transform=CenterCropResizer(config.img_size, config.crop_size))
    eval_sampler = RandSampler(eval_dataset, batch_size=config.batch_size, drop_last=True, shuffle=False)
    eval_loader = DataLoader(eval_dataset, num_workers=4, pin_memory=True, collate_fn=FixCollector, batch_sampler=eval_sampler)

    if is_master:
        crop_source(eval_dataset.images, os.path.join(data_path, 'crop'), transform=CenterCropResizer(config.img_size, config.crop_size))

    if evaluate:
        validate(eval_loader, model, device, ptdtype)
        cal_fid(os.path.join(data_path, 'crop'), os.path.join(cur_dir, 'image'), model, config, ptdtype, device)
        return

    train_dataset = ImagenetDataset(data_path, 'train', transform=RandomCropResizer(config.img_size))
    # train_dataset = CelebADataset(data_path, "train", transform=CenterCropResizer(config.img_size, config.crop_size))
    if distributed:
        train_sampler = DistRandSampler(train_dataset, batch_size=config.batch_size, drop_last=True)
    else:
        train_sampler = RandSampler(train_dataset, batch_size=config.batch_size, drop_last=True)
    train_loader = DataLoader(train_dataset, num_workers=4, pin_memory=True, collate_fn=FixCollector, batch_sampler=train_sampler)

    if is_master:
        print(config)
        for k, v in list(filter(lambda x: x[0][:2] != '__', inspect.getmembers(config))):
            print(f"\t{k}: {v}")
        print()
        # print(model)
        # print()
        total_params, encoder_params, decoder_params, discriminator_params, pl_params = model.get_num_params()
        print(f"total_params: {total_params/1e6:.2f}M, encoder_params: {encoder_params/1e6:.2f}M, decoder_params: {decoder_params/1e6:.2f}M, discriminator_params: {discriminator_params/1e6:.2f}M, perceptual_params: {pl_params/1e6:.2f}M")
        
    optimizer = [
        torch.optim.AdamW(model.get_vae_params(), lr=config.learning_rate, betas=(config.beta1, config.beta2)),
        torch.optim.AdamW(model.get_disc_params(), lr=config.learning_rate, betas=(config.beta1, config.beta2))
    ]
    scaler = GradScaler(enabled=(dtype == 'float16'))

    if compile:
        model = torch.compile(model)
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu], find_unused_parameters=True)

    iter_num = trained_epoch * len(train_dataset) // (config.gpu_num * config.batch_size * config.gradient_accumulation_steps)
    for epoch in range(trained_epoch, config.num_epoch):
        if iter_num >= config.max_iters:
            break

        begin_time = time.time()
        iter_num = train(train_loader, model, device, optimizer, scaler, ptdtype, is_master, config, iter_num)
        train_sampler.shuffle(epoch)
        
        if is_master:
            validate(eval_loader, model, device, ptdtype)
            if epoch % 5 == 0:
                cal_fid(os.path.join(data_path, 'crop'), os.path.join(cur_dir, 'image'), model, config, ptdtype, device)
            print(f'epoch: {epoch}, consume: {time.time() - begin_time:.3f}s')
            if save_model:
                torch.save({'state_dict': model.state_dict()}, model_path)
        
    if is_master:
        cal_fid(os.path.join(data_path, 'crop'), os.path.join(cur_dir, 'image'), model, config, ptdtype, device)

if __name__ == "__main__":
    # mse->mae，基本持平
    # trans_conv改成nearest，参数量增加，耗时增加，效果持平
    # decoder每一段layers增加一层，耗时增加，loss 0.155->0.152
    # encoder每一段layers增加一层，耗时基本不增，loss 0.152->0.151
    # SPP，无效
    # 改变z_channel，1-16之间基本无效，估计是和CelebA数据集基本都是人脸有关
    # img_size 128->176，生成图像混乱，改变z_channel数量无用，kl_weight*2使生成图像质量明显变好，loss 0.174, rec_loss 0.133, kl_loss 0.041
    # channels [64, 64, 128, 256, 256] -> [128, 128, 256, 512, 512]，在CelebA上无收益，说明模型不行，或者CelebA太简单
    # 增加中间层, loss 0.174->0.171
    # 增加中间层, loss 0.171->0.170
    # 增加rear层, no gain
    # remove tanh, perfermance slightly decay
    # attention block，loss 0.169，基本持平
    # res_block shortcut 1->3, no gain
    # weight_decay, train loss不变，test loss上升，kl_loss大幅上涨，fid基本持平, fid=130
    # 改成encoder和decoder的输出channel对称，decoder参数量11.2M->10.12M, fid=133
    # Perceptual loss变种, train loss收敛变慢，test loss不收敛，kl_loss非常大
    # GAN loss, img_size 64, fid 65->36, img_size 176, fid 130->62，细节增强，但会有树根状纹理
    # train epoch 40->60, fid 62->55
    # disc weight 0.5->1.0，无收益
    # g_loss 采用hinge loss, fid62->70
    # weight decay 0.01, fid62->60
    # encoder layer中的降维模块后置， 压缩率改成8，耗时翻倍，显存翻倍
    # decoder layer+1, fid61.5
    # disc的kernel size改成5, fid62->75, kernel size=3, fid62->63.5
    # disc last layer strip=2, 无收益
    # lr减小至百分之一, fid无收益, rec_loss0.141
    # Perceptual loss, fid57->32, 树根状纹理消失，kl_loss略上升0.041->0.046
    # imagenet数据集
    # 为了保证reconstrct的效果，kl_loss应该设置到很小
    config = ImprovedVAE_Imagenet_Config()
    # config = ImprovedVAE_Config()
    gpu_num = config.gpu_num
    load_model = True
    save_model = True
    distributed = True
    compile = True
    evaluate = False
    trained_epoch = 23
    arch = improved_vae
    dtype = "bfloat16"
    data_path = "/home/work/imagenet"
    # data_path = "/home/work/CelebA"
    cur_dir = "/home/work/disk/vision/autoencoder"
    model_path = "checkpoint/improved_vae.pth.tar"
    pl_model_path = "/home/work/disk/vision/classification/checkpoint/resnet18_acc69.3_s.pth.tar"
    if evaluate:
        distributed = False

    if distributed:
        mp.spawn(main, nprocs=gpu_num, args=(gpu_num, config, distributed, evaluate, load_model, save_model, arch, dtype, data_path, cur_dir, model_path, pl_model_path, trained_epoch, compile))
    else:
        main(0, gpu_num, config, distributed, evaluate, load_model, save_model, arch, dtype, data_path, cur_dir, model_path, pl_model_path, trained_epoch, compile)