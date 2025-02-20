import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        # 第一层卷积，步长为stride，保持通道数一致或者改变
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # 第二层卷积
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 用于匹配输入和输出的维度
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out += self.shortcut(x)
        out = F.relu(out)
        return out


class HomoRegressNet(nn.Module):
    def __init__(self, input_feat_size=[32, 32]):
        super().__init__()
        self.feat_h  = input_feat_size[0]
        self.feat_w  = input_feat_size[1]

        # 下采样1/8
        self.stage1 = nn.Sequential(
            ResidualBlock(2, 64, stride=1),
            ResidualBlock(64, 64, stride=1),
            nn.MaxPool2d(2, 2),

            ResidualBlock(64, 256, stride=1),
            ResidualBlock(256, 256, stride=1),
            nn.MaxPool2d(2, 2),

            ResidualBlock(256, 512, stride=1),
            ResidualBlock(512, 512, stride=1),
            nn.MaxPool2d(2, 2)
        )
        
        self.stage2 = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(in_features=512, out_features=2048, bias=True),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=2048, out_features=1024, bias=True),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=1024, out_features=8, bias=True)
        )
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x shape: [N, 2, H, W]
        x = self.stage1(x) # [N, 256, H/4, W/4]
        x = self.stage2(x) # [N, 8]
        return x.reshape(-1, 4, 2)    # [N, 4, 2]

if __name__ == '__main__':
    model = HomoRegressNet()
    feature = torch.rand(1, 2, 32, 32)
    ouput = model(feature)
    print(ouput.shape)