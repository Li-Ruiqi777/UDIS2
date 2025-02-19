import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class SqueezeExcite(nn.Module):
    """轻量级通道注意力模块"""
    def __init__(self, in_chs, reduction_ratio=4):
        super().__init__()
        reduction_chs = max(in_chs // reduction_ratio, 8)
        self.block = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_chs, reduction_chs, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduction_chs, in_chs, 1),
            nn.Hardsigmoid()
        )
    
    def forward(self, x):
        return x * self.block(x)

class ResidualSE(nn.Module):
    """带通道注意力的残差块"""
    def __init__(self, in_chs, expand_chs, kernel_size=3, stride=1):
        super().__init__()
        self.use_res = stride == 1 and in_chs == expand_chs
        
        self.conv = nn.Sequential(
            # 倒置瓶颈结构
            nn.Conv2d(in_chs, expand_chs, 1, bias=False),
            nn.BatchNorm2d(expand_chs),
            nn.ReLU6(inplace=True),
            
            # 深度可分离卷积
            nn.Conv2d(expand_chs, expand_chs, kernel_size, stride, 
                      padding=(kernel_size-1)//2, groups=expand_chs, bias=False),
            nn.BatchNorm2d(expand_chs),
            nn.ReLU6(inplace=True),
            
            SqueezeExcite(expand_chs),  # 通道注意力
            
            # 投影层
            nn.Conv2d(expand_chs, in_chs, 1, bias=False),
            nn.BatchNorm2d(in_chs),
        )
    
    def forward(self, x):
        if self.use_res:
            return x + self.conv(x)
        else:
            return self.conv(x)


class FeatureExtractor(nn.Module):
    def __init__(self, output_scales=[1/4, 1/8, 1/16, 1/32]):
        super().__init__()
        self.stages = nn.ModuleList()
        in_chs = 3
        stem_chs = 16
        self.output_scales = output_scales
        
        # 初始下采样
        self.stem = nn.Sequential(
            nn.Conv2d(in_chs, stem_chs, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem_chs),
            nn.ReLU6(inplace=True)
        )
        
        # 多尺度特征生成（修改配置参数）
        configs = [
            # (expand_ratio, out_chs, repeats, stride, kernel)
            (4,  64, 2, 2, 3),  # stage0: 1/4尺度
            (6,  128, 3, 2, 3),  # stage1: 1/8尺度
            (6,  256, 4, 2, 5),  # stage2: 1/16尺度
            (6, 512, 3, 2, 5),  # stage3: 1/32尺度
        ]
        
        current_chs = stem_chs
        for expand_ratio, out_chs, repeats, stride, kernel in configs:
            layers = []
            
            # 下采样模块（新增通道调整）
            if stride > 1:
                # 通道调整 + 深度可分离下采样
                layers.append(
                    nn.Conv2d(current_chs, out_chs, 1, stride=1, bias=False)
                )
                layers.append(nn.BatchNorm2d(out_chs))
                layers.append(nn.ReLU6(inplace=True))
                
                # 深度卷积实现空间下采样
                layers.append(
                    nn.Conv2d(out_chs, out_chs, kernel, 
                             stride=stride, padding=(kernel//2),
                             groups=out_chs, bias=False)
                )
                layers.append(nn.BatchNorm2d(out_chs))
            
            # 注意力残差模块
            expand_chs = int(out_chs * expand_ratio)
            for _ in range(repeats):
                layers.append(ResidualSE(out_chs, expand_chs, kernel))
            
            self.stages.append(nn.Sequential(*layers))
            current_chs = out_chs  # 更新通道数

    def forward(self, x):
        features = []
        x = self.stem(x)
        
        for i, stage in enumerate(self.stages):
            x = stage(x)
            current_scale = 1 / (2 ** (i + 2))  # stem已下采样2倍
            if current_scale in self.output_scales:
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
    model = FeatureExtractor([1/8, 1/16])
    model = FeatureExtractor_resnet()
    input = torch.randn(1, 3, 512, 512)
    features = model(input)  # 得到4个尺度的特征图

    for feat in features:
        print(feat.shape) 