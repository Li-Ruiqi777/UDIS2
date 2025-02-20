import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from Conv import Conv
from block import C3k2

class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.stage1 = nn.Sequential(
            Conv(3, 64, k=3, s=2), # 1/2
            Conv(64, 128, k=3, s=2), # 1/4
            C3k2(128, 256, 1, False, 0.25),
            Conv(256, 256, k=3, s=2), # 1/8
            C3k2(256, 512, 1, False, 0.25),
        )
        self.stage2 = nn.Sequential(
            Conv(512, 512, k=3, s=2), # 1/16
            C3k2(512, 512, 1, False, 0.25),
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
        features = []
        x = self.stage1(x)
        features.append(x)
        x = self.stage2(x)
        features.append(x)

        return features

class FeatureExtractor_resnet(nn.Module):
    def __init__(self):
        super().__init__()
        resnet50_model = models.resnet.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.feature_extractor_stage1 = nn.Sequential(
            resnet50_model.conv1,
            resnet50_model.bn1,
            resnet50_model.relu,
            resnet50_model.maxpool,
            resnet50_model.layer1,
            resnet50_model.layer2,
        )

        self.feature_extractor_stage2 = nn.Sequential(
            resnet50_model.layer3,
        )

    def forward(self, x):
        features = []
        x = self.feature_extractor_stage1(x)
        features.append(x)
        x = self.feature_extractor_stage2(x)
        features.append(x)
        
        return features

if __name__ == '__main__':
    model = FeatureExtractor()
    # model = FeatureExtractor_resnet()
    input = torch.randn(1, 3, 512, 512)
    features = model(input)  # 得到4个尺度的特征图

    for feat in features:
        print(feat.shape) 