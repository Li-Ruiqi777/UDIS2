import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from ResidualBlock import ResidualBlock
from block import *
    
class MeshRegressNet(nn.Module):
    def __init__(self, grid_size, input_feat_size=[64, 64], spp_pool_size=[1, 2, 4], input_dim=2):
        super().__init__()
        self.grid_h, self.grid_w = grid_size
        self.feat_h  = input_feat_size[0]
        self.feat_w  = input_feat_size[1]
        # 下采样1/16
        self.stage1 = nn.Sequential(
            ResidualBlock(2, 64, stride=2),  # 下采样1/2
            ResidualBlock(64, 64, stride=1),
            ECA(),
            ResidualBlock(64, 128, stride=2),  # 下采样1/4
            ResidualBlock(128, 128, stride=1),
            ECA(),
            ResidualBlock(128, 256, stride=2),  # 下采样1/8
            ResidualBlock(256, 256, stride=1),
            ECA(),
            ResidualBlock(256, 512, stride=2),  # 下采样1/16
            ResidualBlock(512, 512, stride=1),
            ECA(),
        )

        self.stage2 = nn.Sequential(
            SPP(spp_pool_size),
            nn.Linear(in_features=512 * sum([p*p for p in spp_pool_size]), out_features=4096, bias=True),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=4096, out_features=2048, bias=True),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=2048, out_features=(self.grid_w+1)*(self.grid_h+1)*2, bias=True)
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
        x = self.stage1(x)
        x = self.stage2(x)
        return x.reshape(-1, self.grid_h + 1, self.grid_w + 1, 2)    # [N, H, W, 2]

if __name__ == '__main__':
    model = MeshRegressNet([12, 12],[64, 64])
    feature = torch.rand(1, 2, 64, 64)
    ouput = model(feature)
    print(ouput.shape)