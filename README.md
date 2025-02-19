#UDIS2

## TODO

- [ ] 重新搞个BackBone，还是先H再Mesh但不用ResNet，重新封装个类
- [ ] 看论文了解注意力机制一般加载哪里？加几个?



## 开发中遇到的问题

1.如何在主干网络加注意力机制？

目前的主干网络是ResNet，在它里面加注意力机制其实有2个思路：

- 加在残差块里面（也就是`resnet.py`里的`BasicBlock`类或者`BottleNeck`类）
- 加在残差块之间

ResNet其实不同变种的基本结构类似，包含的几个大层都是固定的，只是里面的残差块的数目不同

```python
def _forward_impl(self, x: Tensor) -> Tensor:
    # See note [TorchScript super()]
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    x = self.avgpool(x)
    x = torch.flatten(x, 1)
    x = self.fc(x)

    return x

```

因此，可以把注意力机制加在几个大层(layer1，layer2...)之间

> 由于使用方法二的计算量更少，且可以使用预训练权重，所以采用方法二

注意：修改代码时，可以直接改`resnet.py`，也可以修改UDIS2的模型定义，我选择的是后者

参考链接：

- [pytorch中加入注意力机制（CBAM），以ResNet为例。解析到底要不要用ImageNet预训练？如何加预训练参数？ - 知乎](https://zhuanlan.zhihu.com/p/99261200)
- [【深度学习】Pytorch：在 ResNet 中加入注意力机制_注意力机制resnet-CSDN博客](https://blog.csdn.net/2303_80346267/article/details/145226698)
- [HAM:Hybrid attention module神经网络中混合注意力模块代码全网首次开源复现 - 知乎](https://zhuanlan.zhihu.com/p/555252748)



2.注意力机制要加几个呢？
