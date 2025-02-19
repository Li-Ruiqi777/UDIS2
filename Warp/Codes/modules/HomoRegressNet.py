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
    def __init__(self, input_channels=2, feat_dim=256):
        super().__init__()
        
        # 特征提取主干
        self.feature_net = nn.Sequential(
            nn.Conv2d(input_channels, 64, 7, stride=2, padding=3),  # 1/2
            nn.BatchNorm2d(64),
            nn.ReLU(),
            ResBlock(64),
            nn.Conv2d(64, 128, 5, stride=2, padding=2),  # 1/4
            nn.BatchNorm2d(128),
            ResBlock(128),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),  # 1/8
            nn.BatchNorm2d(256),
            ResBlock(256),
            ChannelAttention(256),
            nn.AdaptiveAvgPool2d(1)  # 全局特征
        )
        
        # 回归头
        self.regressor = nn.Sequential(
            nn.Linear(256, feat_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(feat_dim, 8)  # 输出8个参数
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
        features = self.feature_net(x)  # [N, 256, 1, 1]
        features = features.view(features.size(0), -1)  # [N, 256]
        params = self.regressor(features)  # [N, 8]
        return params.reshape(-1, 4, 2)    # [N, 4, 2]

if __name__ == '__main__':
    model = HomoRegressNet()
    feature = torch.rand(1, 2, 16, 16)
    ouput = model(feature)
    print(ouput.shape)