import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    """轻量级通道注意力"""
    def __init__(self, in_planes, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(in_planes, in_planes // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_planes // reduction, in_planes),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        out = avg_out + max_out
        return x * out.view(b, c, 1, 1)

class ResBlock(nn.Module):
    """带注意力的残差块"""
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.ca = ChannelAttention(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_channels)
        
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.ca(out)
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

class HomoRegressNet(nn.Module):
    def __init__(self, input_feat_size=[32, 32]):
        super().__init__()
        self.feat_h  = input_feat_size[0]
        self.feat_w  = input_feat_size[1]

        # 下采样1/8
        self.stage1 = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        self.stage2 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=256 * self.feat_h//8 * self.feat_w//8, out_features=4096, bias=True),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=4096, out_features=1024, bias=True),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=1024, out_features=8, bias=True)
        )
        
        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
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