import torch
import torch.nn as nn
import torch.nn.functional as functional

class Conv_Batch_Active(nn.Module):
    def __init__(self, cin, out, kernel, stride=1, padding=0, bias=True, bn=False, active=True):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(cin, out, kernel, stride, padding, bias=bias),
            nn.BatchNorm2d(out) if bn else nn.Identity(),
            nn.ReLU(inplace=True) if active else nn.Identity(),
        )

    def forward(self, x):
        return self.block(x)

class BasicBlock(nn.Module):
    def __init__(self, cin, out, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            Conv_Batch_Active(cin, out, 3, stride, 1, bn=True),
            Conv_Batch_Active(out, out, 3, 1, 1, bn=True, active=False)
        )

        self.shortcut = nn.Identity() if cin == out and stride == 1 else \
                        Conv_Batch_Active(cin, out, 1, stride, bn=True, active=False)

    def forward(self, x):
        return functional.relu(self.block(x) + self.shortcut(x), inplace=True)

class ResNet(nn.Module):
    def __init__(self, block, layers, channels, test=False):
        super(ResNet, self).__init__()
        self.test = test
        self.conv1 = Conv_Batch_Active(3, 64, 7, 2, 3, bn=True)
        self.conv2 = self._make_layer(block, layers[0], channels[0], channels[1], maxpool=True)
        self.conv3 = self._make_layer(block, layers[1], channels[1], channels[2])
        self.conv4 = self._make_layer(block, layers[2], channels[2], channels[3])
        self.conv5 = self._make_layer(block, layers[3], channels[3], channels[4])
        if test:
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.linear = nn.Linear(channels[4], 1000)

    @staticmethod
    def _make_layer(block, num_layer, cin, out, maxpool=False):
        if maxpool:
            layers = [nn.MaxPool2d(3, 2, 1),
                      block(cin, out)]
        else:
            layers = [block(cin, out, 2)]
        for i in range(num_layer-1):
            layers.append(block(out, out))
        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        if self.test:
            y = self.pool(x5)
            y = torch.flatten(y, 1)
            y = self.linear(y)
            return y
        else:
            return x2, x3, x4, x5

def resnet18(pretrained=False, model_path=None, test=False):
    model = ResNet(BasicBlock, [2, 2, 2, 2], [64, 64, 128, 256, 512], test=test)
    if pretrained:
            state_dict = torch.load(model_path)['state_dict']
            model.load_state_dict({k.replace('module.', ''): v for k, v in state_dict.items()}, strict=test)
    return model

if __name__ == '__main__':
    x = resnet18()
    print(x)