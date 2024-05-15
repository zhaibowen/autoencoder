import torch
import torch.nn as nn

def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(nn.functional.relu(1. - logits_real))
    loss_fake = torch.mean(nn.functional.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss

class Conv_Batch_Active(nn.Module):
    def __init__(self, cin, out, kernel, stride=1, padding=0, bias=True, bn=False, active=True):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(cin, out, kernel, stride, padding, bias=bias),
            nn.BatchNorm2d(out) if bn else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True) if active else nn.Identity(),
        )

    def forward(self, x):
        return self.block(x)

class Discriminator(nn.Module):
    def __init__(self, channels=[64, 128, 256, 512]):
        super().__init__()
        self.conv1 = Conv_Batch_Active(3, channels[0], 4, 2, 1, bn=True)
        self.conv2 = Conv_Batch_Active(channels[0], channels[1], 4, 2, 1, bn=True)
        self.conv3 = Conv_Batch_Active(channels[1], channels[2], 4, 2, 1, bn=True)
        self.conv4 = Conv_Batch_Active(channels[2], channels[3], 4, 1, 1, bn=True)
        self.conv5 = nn.Conv2d(channels[3], 1, 4, 1, 1)

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

if __name__ == "__main__":
    x = Discriminator()
    y = torch.zeros([10,3,64,64])
    z = x(y)