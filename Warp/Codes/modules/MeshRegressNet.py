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
    
class MeshRegressNet(nn.Module):
    def __init__(self, grid_size, input_feat_size=[64, 64], input_dim=2):
        super().__init__()
        self.grid_h, self.grid_w = grid_size
        self.feat_h  = input_feat_size[0]
        self.feat_w  = input_feat_size[1]
        # 下采样1/16
        self.stage1 = nn.Sequential(
            ResidualBlock(2, 64, stride=1),
            ResidualBlock(64, 64, stride=1),
            nn.MaxPool2d(2, 2),

            ResidualBlock(64, 128, stride=1),
            ResidualBlock(128, 128, stride=1),
            nn.MaxPool2d(2, 2),

            ResidualBlock(128, out_channels=256, stride=1),
            ResidualBlock(256, 256, stride=1),
            nn.MaxPool2d(2, 2),

            ResidualBlock(256, 512, stride=1),
            ResidualBlock(512, 512, stride=1),
            nn.MaxPool2d(2, 2),
        )

        self.stage2 = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(in_features=512, out_features=4096, bias=True),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=4096, out_features=2048, bias=True),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=2048, out_features=(self.grid_w+1)*(self.grid_h+1)*2, bias=True)
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
        x = self.stage1(x)
        x = self.stage2(x)
        return x.reshape(-1, self.grid_h + 1, self.grid_w + 1, 2)    # [N, H, W, 2]

if __name__ == '__main__':
    model = MeshRegressNet([12, 12],[64, 64])
    feature = torch.rand(1, 2, 64, 64)
    ouput = model(feature)
    print(ouput.shape)