import torch
import torch.nn as nn
import torch.nn.functional as F

class MeshRegressNet(nn.Module):
    def __init__(self, grid_size, input_feat_size=[64, 64], input_dim=2):
        super().__init__()
        self.grid_h, self.grid_w = grid_size
        self.feat_h  = input_feat_size[0]
        self.feat_w  = input_feat_size[1]
        # 下采样1/16
        self.stage1 = nn.Sequential(
            nn.Conv2d(input_dim, 64, kernel_size=3, padding=1, bias=False),
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
            nn.MaxPool2d(2, 2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        self.stage2 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=512 * self.feat_h//16 * self.feat_w//16, out_features=4096, bias=True),
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

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        return x.reshape(-1, self.grid_h + 1, self.grid_w + 1, 2)    # [N, H, W, 2]

if __name__ == '__main__':
    model = MeshRegressNet([12, 12],[64, 64])
    feature = torch.rand(1, 2, 64, 64)
    ouput = model(feature)
    print(ouput.shape)