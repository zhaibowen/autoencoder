import torch
import torch.nn as nn
from .AutoEncoder import Encoder, Decoder

class DiagonalGaussianDistribution(object):
    def __init__(self, parameters):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)

    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape).to(device=self.parameters.device)
        return x

    def kl(self):
        return 0.5 * torch.sum(torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar, dim=[1, 2, 3])

class AutoEncoderKL(nn.Module):
    def __init__(self, config, flash=False):
        super().__init__()
        self.encoder = Encoder(config.layers, config.channels, 2*config.z_channel, flash)
        self.quant_conv = torch.nn.Conv2d(2*config.z_channel, 2*config.embed_dim, 1)

        self.post_quant_conv = torch.nn.Conv2d(config.embed_dim, config.z_channel, 1)
        self.decoder = Decoder(config.layers, config.channels, config.z_channel, flash)

    def get_num_params(self):
        encoder_params = self.encoder.get_num_params()
        decoder_params = self.decoder.get_num_params()
        n_params = sum(p.numel() for p in self.parameters())
        return n_params, encoder_params, decoder_params

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, input):
        posterior = self.encode(input)
        z = posterior.sample()
        dec = self.decode(z)
        return dec, posterior

def autoencoder_kl60(config, pretrained=False, model_path=None, flash=False):
    # encoder layers 26 + decoder layers 34
    model = AutoEncoderKL(config, flash)
    if pretrained:
        state_dict = torch.load(model_path)['state_dict']
        model.load_state_dict({k.replace('module._orig_mod.', ''): v for k, v in state_dict.items()})
    return model

if __name__ == "__main__":
    layers = [2, 2, 2, 2]
    # channels = [128, 128, 256, 512, 512] # 84M
    channels = [64, 64, 128, 256, 256] # 21M
    z_channel = 4
    embed_dim = 4

    x = AutoEncoderKL(layers, channels, z_channel, embed_dim)
    print(x.get_num_params())
    y = torch.zeros([10,3,256,256])
    z = x(y)