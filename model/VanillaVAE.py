import torch
import torch.nn as nn

class Conv_Batch_Active(nn.Module):
    def __init__(self, cin, out, kernel, stride=1, padding=0, bias=True, bn=False, active=True):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(cin, out, kernel, stride, padding, bias=bias),
            nn.BatchNorm2d(out) if bn else nn.Identity(),
            nn.LeakyReLU(inplace=True) if active else nn.Identity(),
        )

    def forward(self, x):
        return self.block(x)

class TransConv_Batch_Active(nn.Module):
    def __init__(self, cin, out, kernel, stride=1, padding=0, bias=True, bn=False, active=True):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(cin, out, kernel, stride, padding, output_padding=1, bias=bias),
            nn.BatchNorm2d(out) if bn else nn.Identity(),
            nn.LeakyReLU(inplace=True) if active else nn.Identity(),
        )

    def forward(self, x):
        return self.block(x)

class Encoder(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = Conv_Batch_Active(3, channels[0], 3, 2, 1, bn=True)
        self.conv2 = Conv_Batch_Active(channels[0], channels[1], 3, 2, 1, bn=True)
        self.conv3 = Conv_Batch_Active(channels[1], channels[2], 3, 2, 1, bn=True)
        self.conv4 = Conv_Batch_Active(channels[2], channels[3], 3, 2, 1, bn=True)
        self.conv5 = Conv_Batch_Active(channels[3], channels[4], 3, 2, 1, bn=True)

    def get_num_params(self):
        n_params = sum(p.numel() for p in self.parameters())
        return n_params
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x

class Decoder(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = TransConv_Batch_Active(channels[4], channels[3], 3, 2, 1, bn=True)
        self.conv2 = TransConv_Batch_Active(channels[3], channels[2], 3, 2, 1, bn=True)
        self.conv3 = TransConv_Batch_Active(channels[2], channels[1], 3, 2, 1, bn=True)
        self.conv4 = TransConv_Batch_Active(channels[1], channels[0], 3, 2, 1, bn=True)

    def get_num_params(self):
        n_params = sum(p.numel() for p in self.parameters())
        return n_params
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x

class VanillaVAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = Encoder(config.channels)
        self.fc_mu = nn.Linear(config.channels[-1] * 4, config.latent_dim)
        self.fc_var = nn.Linear(config.channels[-1] * 4, config.latent_dim)

        self.decoder_input = nn.Linear(config.latent_dim, config.channels[-1] * 4)
        self.decoder = Decoder(config.channels)

        self.final_layer = nn.Sequential(
            TransConv_Batch_Active(config.channels[0], config.channels[0], 3, 2, 1, bn=True),
            nn.Conv2d(config.channels[0], out_channels=3, kernel_size= 3, padding= 1),
            nn.Tanh()
        )

    def get_num_params(self):
        encoder_params = self.encoder.get_num_params()
        decoder_params = self.decoder.get_num_params()
        n_params = sum(p.numel() for p in self.parameters())
        return n_params, encoder_params, decoder_params
    
    def encode(self, input):
        # 3, 64, 64
        result = self.encoder(input)
        # 512, 2, 2
        result = torch.flatten(result, start_dim=1)
        # 2048
        mu = self.fc_mu(result)
        # 128
        log_var = self.fc_var(result)
        # 128
        return [mu, log_var]
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def decode(self, z):
        result = self.decoder_input(z)
        result = result.view(-1, self.config.channels[-1], 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def forward(self, input):
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        rec = self.decode(z)
        return rec, mu, log_var
    
    def loss(self, input, recons, mu, log_var):
        recons_loss = nn.functional.mse_loss(recons, input)
        kl_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        loss = recons_loss + self.config.kl_weight * kl_loss
        return recons_loss, self.config.kl_weight * kl_loss, loss
    
def vanilla_vae(config, pretrained=False, model_path=None):
    model = VanillaVAE(config)
    if pretrained:
        state_dict = torch.load(model_path)['state_dict']
        model.load_state_dict({k.replace('module._orig_mod.', ''): v for k, v in state_dict.items()})
    return model