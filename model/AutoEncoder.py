import math
import torch
import torch.nn as nn

class Norm_Active_Conv(nn.Module):
    def __init__(self, cin, out, num_groups=32):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(num_groups=num_groups, num_channels=cin, eps=1e-6),
            nn.SiLU(inplace=True),
            nn.Conv2d(cin, out, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        return self.block(x)

class Downsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        x = nn.functional.pad(x, (0, 1, 0, 1), mode="constant", value=0)
        x = self.conv(x)
        return x

class Upsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        x = self.conv(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, cin, out):
        super().__init__()
        self.block = nn.Sequential(
            Norm_Active_Conv(cin, out),
            Norm_Active_Conv(out, out)
        )

        self.shortcut = nn.Identity() if cin == out else \
                        nn.Conv2d(cin, out, kernel_size=1)

    def forward(self, x):
        return self.block(x) + self.shortcut(x)

class AttnBlock(nn.Module):
    def __init__(self, in_channels, flash):
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
    def __init__(self, layers, channels, z_channel, flash=False):
        super().__init__()
        self.conv1 = nn.Conv2d(3, channels[0], kernel_size=3, stride=1, padding=1)
        self.conv2 = self._make_layer(layers[0], channels[0], channels[1], down_sample=True)
        self.conv3 = self._make_layer(layers[1], channels[1], channels[2], down_sample=True)
        self.conv4 = self._make_layer(layers[2], channels[2], channels[3], down_sample=True)
        self.conv5 = self._make_layer(layers[3], channels[3], channels[4])

        cin = channels[4]
        self.out = nn.Sequential(
            ResnetBlock(cin, cin),
            AttnBlock(cin, flash),
            ResnetBlock(cin, cin),
            Norm_Active_Conv(cin, z_channel)
        )

    def get_num_params(self):
        n_params = sum(p.numel() for p in self.parameters())
        return n_params
    
    @staticmethod
    def _make_layer(num_layer, cin, out, down_sample=False):
        layers = []
        for _ in range(num_layer):
            layers.append(ResnetBlock(cin, out))
            cin = out
        if down_sample:
            layers.append(Downsample(out))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.out(x)
        return x

class Decoder(nn.Module):
    def __init__(self, layers, channels, z_channel, flash=False):
        super().__init__()
        cin = channels[4]
        self.conv1 = nn.Conv2d(z_channel, cin, kernel_size=3, stride=1, padding=1)

        self.mid = nn.Sequential(
            ResnetBlock(cin, cin),
            AttnBlock(cin, flash),
            ResnetBlock(cin, cin),
        )

        self.conv2 = self._make_layer(layers[3], channels[4], channels[4], up_sample=True)
        self.conv3 = self._make_layer(layers[2], channels[4], channels[3], up_sample=True)
        self.conv4 = self._make_layer(layers[1], channels[3], channels[2], up_sample=True)
        self.conv5 = self._make_layer(layers[0], channels[2], channels[1])

        self.out = Norm_Active_Conv(channels[1], 3)

    def get_num_params(self):
        n_params = sum(p.numel() for p in self.parameters())
        return n_params
    
    @staticmethod
    def _make_layer(num_layer, cin, out, up_sample=False):
        layers = []
        for _ in range(num_layer+1):
            layers.append(ResnetBlock(cin, out))
            cin = out
        if up_sample:
            layers.append(Upsample(out))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.mid(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.out(x)
        return x

if __name__ == "__main__":
    layers = [2, 2, 2, 2]
    channels = [128, 128, 256, 512, 512]
    z_channel = 4

    encoder = Encoder(layers, channels, z_channel) # *2
    decoder = Decoder(layers, channels, z_channel)
    
    print(encoder.get_num_params())
    print(decoder.get_num_params())

    y = torch.zeros([10,3,256,256])
    z = encoder(y)
    z = decoder(z)