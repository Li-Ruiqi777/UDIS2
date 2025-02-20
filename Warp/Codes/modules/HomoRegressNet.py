import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.ResidualBlock import ResidualBlock

class HomoRegressNet(nn.Module):
    def __init__(self, input_feat_size=[32, 32]):
        super().__init__()
        self.feat_h  = input_feat_size[0]
        self.feat_w  = input_feat_size[1]

        # 下采样1/8
        self.stage1 = nn.Sequential(
            ResidualBlock(2, 64, stride=2),  # 下采样1/2
            ResidualBlock(64, 64, stride=1),
            ResidualBlock(64, 128, stride=2),  # 下采样1/4
            ResidualBlock(128, 128, stride=1),
            ResidualBlock(128, 256, stride=2),  # 下采样1/8
            ResidualBlock(256, 256, stride=1),
        )
        
        self.stage2 = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            # nn.Linear(in_features=256 * self.feat_h//8 * self.feat_w//8, out_features=2048, bias=True),
            nn.Linear(in_features=256 * 4 * 4, out_features=2048, bias=True),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=2048, out_features=1024, bias=True),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=1024, out_features=8, bias=True)
        )
        
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
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