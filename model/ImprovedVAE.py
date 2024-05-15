import math
import torch
import torch.nn as nn
from .discriminator import Discriminator, hinge_d_loss
from .lpips import LPIPS

class Conv_Batch_Active(nn.Module):
    def __init__(self, cin, out, kernel, stride=1, padding=0, bias=True, bn=False, active=True, trans=False):
        super().__init__()
        if trans:
            module = nn.ConvTranspose2d(cin, out, kernel, stride, padding, output_padding=1, bias=bias)
        else:
            module = nn.Conv2d(cin, out, kernel, stride, padding, bias=bias)

        self.block = nn.Sequential(
            module,
            nn.BatchNorm2d(out) if bn else nn.Identity(),
            nn.SiLU(inplace=True) if active else nn.Identity(),
        )

    def forward(self, x):
        return self.block(x)

class ResBlock(nn.Module):
    def __init__(self, cin, out, stride=1, trans=False):
        super().__init__()
        self.block = nn.Sequential(
            Conv_Batch_Active(cin, out, 3, stride, 1, bn=True, trans=trans),
            Conv_Batch_Active(out, out, 3, 1, 1, bn=True, active=False)
        )

        self.shortcut = nn.Identity() if cin == out and stride == 1 else \
                        Conv_Batch_Active(cin, out, 1, stride, bn=True, active=False, trans=trans)

    def forward(self, x):
        return nn.functional.silu(self.block(x) + self.shortcut(x), inplace=True)

class AttnBlock(nn.Module):
    def __init__(self, in_channels, flash=False):
        super().__init__()
        self.flash = flash
        self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6)
        self.q_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.k_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.v_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.o_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        hidden_states = self.norm(x)
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        batch_size, channels, height, width = query.shape
        query = query.reshape(batch_size, channels, height*width).transpose(1, 2)   # b, hw, c
        key = key.reshape(batch_size, channels, height*width).transpose(1, 2) # b, hw, c
        value = value.reshape(batch_size, channels, height*width).transpose(1, 2) # b, hw, c

        if self.flash:
            output = nn.functional.scaled_dot_product_attention(query, key, value)
        else:
            att = torch.matmul(query, key.transpose(1, 2)) / math.sqrt(channels) # b, hw, hw
            att = nn.functional.softmax(att, dim=-1)
            output = torch.matmul(att, value) # b, hw, c

        output = output.transpose(1, 2).reshape(batch_size, channels, height, width).contiguous() # b, c, h, w
        output = self.o_proj(output)

        return x + output

class Encoder(nn.Module):
    def __init__(self, layers, channels, z_channel):
        super().__init__()
        self.conv1 = Conv_Batch_Active(3, channels[0], 3, 1, 1, bn=True)
        self.conv2 = self._make_layer(layers[0], channels[0], channels[1])
        self.conv3 = self._make_layer(layers[1], channels[1], channels[2])
        self.conv4 = self._make_layer(layers[2], channels[2], channels[3])
        self.conv5 = self._make_layer(layers[3], channels[3], channels[4])
        self.conv6 = nn.Sequential(
            ResBlock(channels[4], channels[4]),
            AttnBlock(channels[4]),
            ResBlock(channels[4], channels[4]),
            Conv_Batch_Active(channels[4], z_channel*2, 3, 1, 1, bn=False, active=False)
        )

    def get_num_params(self):
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    @staticmethod
    def _make_layer(num_layer, cin, out):
        layers = [ResBlock(cin, out, 2)]
        for i in range(num_layer-1):
            layers.append(ResBlock(out, out))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        return x

class Decoder(nn.Module):
    def __init__(self, layers, channels, z_channel):
        super().__init__()   
        self.conv1 = nn.Sequential(
            Conv_Batch_Active(z_channel, channels[4], 3, 1, 1, bn=True),
            ResBlock(channels[4], channels[4]),
            AttnBlock(channels[4]),
            ResBlock(channels[4], channels[4]),
        )
        self.conv2 = self._make_layer(layers[3], channels[4], channels[3])
        self.conv3 = self._make_layer(layers[2], channels[3], channels[2])
        self.conv4 = self._make_layer(layers[1], channels[2], channels[1])
        self.conv5 = self._make_layer(layers[0], channels[1], channels[0])
        self.conv6 = Conv_Batch_Active(channels[0], 3, 3, 1, 1, bn=False, active=False)

    def get_num_params(self):
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    @staticmethod
    def _make_layer(num_layer, cin, out):
        layers = [ResBlock(cin, cin, 2, trans=True)]
        for i in range(num_layer-1):
            if i < num_layer - 2:
                layers.append(ResBlock(cin, cin))
            else:
                layers.append(ResBlock(cin, out))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        return x
    
class ImprovedVAE(nn.Module):
    def __init__(self, config, pl_model_path=None):
        super().__init__()
        self.config = config
        self.encoder = Encoder(config.layers, config.channels, config.z_channel)
        self.decoder = Decoder(config.layers, config.channels, config.z_channel)
        self.final_layer = nn.Tanh()
        
        self.discriminator = Discriminator()
        self.last_layer = self.decoder.conv6.block[0].weight

        self.pl_model = LPIPS(model_path=pl_model_path) if pl_model_path else None

    def get_vae_params(self):
        return list(self.encoder.parameters())+ list(self.decoder.parameters())

    def get_disc_params(self):
        return self.discriminator.parameters()

    def get_num_params(self):
        encoder_params = self.encoder.get_num_params()
        decoder_params = self.decoder.get_num_params()
        discriminator_params = self.discriminator.get_num_params()
        pl_params = self.pl_model.get_num_params()
        n_params = sum(p.numel() for p in self.parameters())
        return n_params, encoder_params, decoder_params, discriminator_params, pl_params

    def encode(self, input):
        # 3, 64, 64
        result = self.encoder(input)
        # 1024, 4, 4
        mu, log_var = torch.chunk(result, 2, dim=1)
        log_var = torch.clamp(log_var, -30.0, 20.0)
        # 512, 4, 4
        return [mu, log_var]
    
    def reparameterize(self, mu, logvar, sample_posterior=True):
        if sample_posterior == False:
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def decode(self, z):
        # 512, 4, 4
        result = self.decoder(z)
        # 3, 64, 64
        result = self.final_layer(result)
        return result

    def forward(self, input, optimizer_idx=None, need_g_loss=False, sample_posterior=True):
        # optimizer_idx None: infer, 0: 计算vae的loss，1: 计算disc的loss, 2: valid mode, 都计算
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var, sample_posterior)
        rec = self.decode(z)

        if optimizer_idx is None:
            return rec
        
        rec_loss, kl_loss, pl_loss, loss, d_weight, g_loss, d_loss = [0] * 7
        if optimizer_idx == 0 or optimizer_idx == 2:
            rec_loss, kl_loss, pl_loss, g_loss, loss, d_weight = self.loss(input, rec, mu, log_var, need_g_loss, optimizer_idx!=2)
        if optimizer_idx == 1 or optimizer_idx == 2:
            d_loss = self.disc_loss(input, rec)
        return rec_loss, kl_loss, pl_loss, loss, d_weight, g_loss, d_loss
    
    def loss(self, input, rec, mu, log_var, need_g_loss=False, need_d_weight=True):
        rec_loss = nn.functional.l1_loss(rec, input)
        kl_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=[1, 2, 3]), dim = 0) * self.config.kl_weight
        pl_loss = self.pl_model(input, rec) * self.config.pl_weight

        g_loss, d_weight = torch.zeros(1).to(kl_loss.device), 1.0
        if need_g_loss:
            g_loss = torch.mean(nn.functional.relu(-self.discriminator(rec)))
            if need_d_weight:
                d_weight = self.calculate_adaptive_weight(rec_loss, g_loss)
            g_loss = g_loss * d_weight
        
        loss = rec_loss + kl_loss + pl_loss + g_loss
        return rec_loss, kl_loss, pl_loss, g_loss, loss, d_weight
    
    def disc_loss(self, input, rec):
        logits_real = self.discriminator(input.detach())
        logits_fake = self.discriminator(rec.detach())
        d_loss = hinge_d_loss(logits_real, logits_fake)
        return d_loss
    
    def calculate_adaptive_weight(self, rec_loss, g_loss):
        rec_grads = torch.autograd.grad(rec_loss, self.last_layer, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, self.last_layer, retain_graph=True)[0]

        d_weight = torch.linalg.norm(rec_grads) / (torch.linalg.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        return d_weight * self.config.disc_weight
    
def improved_vae(config, pretrained=False, model_path=None, pl_model_path=None):
    model = ImprovedVAE(config, pl_model_path)
    if pretrained:
        state_dict = torch.load(model_path)['state_dict']
        replacer = 'module.'
        if "module._orig_mod." == list(state_dict.keys())[0][:17]:
            replacer = 'module._orig_mod.'
        model.load_state_dict({k.replace(replacer, ''): v for k, v in state_dict.items()}, strict=False)
    return model

if __name__ == "__main__":
    from dataclasses import dataclass
    @dataclass
    class ImprovedVAE_Config:
        layers = [3, 3, 3, 3]
        channels = [64, 64, 128, 256, 256]
        z_channel = 8
        kl_weight = 0.00025

    config = ImprovedVAE_Config()
    x = ImprovedVAE(config)
    print(x)
    # print(x.get_num_params())
    # y = torch.zeros([10,3,64,64])
    # z = x(y)