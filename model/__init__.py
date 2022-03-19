# This project is manually forked from this project: https://github.com/HowieMa/NSRMhand
import os, sys
currentUrl = os.path.dirname(__file__)
parentUrl = os.path.abspath(os.path.join(currentUrl, os.pardir))
sys.path.append(parentUrl)
from .mish import Mish
from .attention_augmentation2D import *
from .blur import *

config = {"ARB": "ARB_add"}
# DepthwiseConv2d: https://gist.github.com/bdsaglam/b16de6ae6662e7a783e06e58e2c5185a
class DepthwiseConv2d(nn.Conv2d):
    def __init__(self,
                 in_channels,
                 depth_multiplier=1,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 dilation=1,
                 bias=True,
                 padding_mode='zeros'
                 ):
        out_channels = in_channels * depth_multiplier
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=bias,
            padding_mode=padding_mode
        )

def aug_block(in_channels, out_channels, kernel_size, dk,dv, Nh, shape):
    return nn.Sequential(
            AugmentedConv(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, dk=dk,
                          dv=dv, Nh=Nh,
                          relative=True, stride=1, shape=shape),
            nn.BatchNorm2d(out_channels),
            Mish()
        )

# torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
class ARB_Add(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, aug=True, dk=0.1, dv=0.1, Nh=4, shape = 224):
        super(ARB_Add, self).__init__()
        self.kernel_size = kernel_size
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * 4, kernel_size=1, padding = 1//2),
            nn.BatchNorm2d(out_channels * 4),
            Mish(),
            DepthwiseConv2d(in_channels=out_channels * 4, kernel_size=kernel_size, padding = (kernel_size-1)//2),
            nn.BatchNorm2d(out_channels * 4),
            Mish()
        )
        self.aug = aug
        if self.aug:
            self.attention_aug = aug_block(out_channels * 4, out_channels * 4, kernel_size, dk, dv, Nh, shape)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels * 4, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            Mish(),
        )

    def forward(self, inputs):
        if not (self.kernel_size % 2):
            inputs = torch.nn.functional.pad(inputs, (0, 1, 0, 1), mode='constant', value=0)
        x = self.conv1(inputs)
        if self.aug:
            augmented_conv = self.attention_aug(x)
            x = torch.add(augmented_conv, x)
        x = self.conv2(x)
        return x

class ARB_Cat(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, aug=True, dk=0.1, dv=0.1, Nh=4, shape = 224):
        super(ARB_Cat, self).__init__()
        self.kernel_size = kernel_size
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * 4, 1, padding = 1//2),
            nn.BatchNorm2d(out_channels * 4),
            Mish(),
            DepthwiseConv2d(in_channels=out_channels * 4, kernel_size=kernel_size, padding = (kernel_size-1)//2),
            nn.BatchNorm2d(out_channels * 4),
            Mish()
        )
        self.aug = aug
        self.attention_aug = aug_block(out_channels * 4, out_channels * 4, kernel_size, dk, dv, Nh, shape)
        if self.aug:
            self.conv2 = nn.Sequential(
                nn.Conv2d(out_channels * 8, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels),
                Mish(),
            )
        else:
            self.conv2 = nn.Sequential(
                nn.Conv2d(out_channels * 4, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels),
                Mish(),
            )

    def forward(self, inputs):
        if not (self.kernel_size % 2):
            inputs = torch.nn.functional.pad(inputs, (0, 1, 0, 1), mode='constant', value=0)
        x = self.conv1(inputs)
        if self.aug:
            augmented_conv = self.attention_aug(x)
            x = torch.cat([augmented_conv, x], dim = 1)
        x = self.conv2(x)
        return x

class ARB_None(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, aug=True, dk=0.1, dv=0.1, Nh=4, shape = 224):
        super(ARB_None, self).__init__()
        self.kernel_size = kernel_size
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * 4, 1, padding = 1//2),
            nn.BatchNorm2d(out_channels * 4),
            Mish(),
            DepthwiseConv2d(in_channels=out_channels * 4, kernel_size=kernel_size, padding = (kernel_size-1)//2),
            nn.BatchNorm2d(out_channels * 4),
            Mish()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels * 4, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            Mish(),
        )

    def forward(self, inputs):
        if not (self.kernel_size % 2):
            inputs = torch.nn.functional.pad(inputs, (0, 1, 0, 1), mode='constant', value=0)
        x = self.conv1(inputs)
        x = self.conv2(x)
        return x


class Dense(nn.Module):
    def __init__(self, in_channels, growth_rate, kernel_size, iteration, Nh=4, aug=True, shape = 224):
        super(Dense, self).__init__()
        self.iteration = iteration
        ARB = eval(config['ARB'])
        self.arb = torch.nn.ModuleList([ARB(in_channels, growth_rate, kernel_size=kernel_size, aug=aug, dk=0.1,
                                dv=0.1, Nh=Nh, shape = shape)])
        for i in range(1, self.iteration):
            self.arb.append(ARB(in_channels + growth_rate*i, growth_rate, kernel_size=kernel_size, aug=aug, dk=0.1,
                                dv=0.1, Nh=Nh, shape = shape))

    def forward(self, inputs):
        x_list = [inputs]
        for i in range(self.iteration):
            inputs = self.arb[i](inputs)
            x_list.append(inputs)
            inputs = torch.cat(x_list, dim=1)
        return inputs


class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Transition, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.activation = BlurPool(out_channels)
        self.batch_normalization = nn.BatchNorm2d(out_channels)

    def forward(self, inputs):
        x = self.conv(inputs)
        x = self.activation(x)
        x = self.batch_normalization(x)
        return x


class light_Model(nn.Module):
    def __init__(self, conf):
        super(light_Model, self).__init__()
        global config
        config = conf
        input_shape = 224
        input_features = 3
        self.dense1 = Dense(in_channels=input_features, growth_rate=10, kernel_size=5, iteration=8, Nh=4, aug=False, shape = input_shape)
        ############################### 112
        self.transition1 = Transition(in_channels=80 + input_features, out_channels=64)
        self.dense2 = Dense(in_channels=64, growth_rate=10, kernel_size=5, iteration=8, Nh=4, aug=False, shape = input_shape//2)
        ############################### 56
        self.transition2 = Transition(in_channels=80 + 64, out_channels=64)
        self.dense3 = Dense(in_channels=64, growth_rate=10, kernel_size=3, iteration=6, Nh=1, aug=True, shape = input_shape//4)
        ############################### 28
        self.transition3 = Transition(in_channels=60 + 64, out_channels=64)
        self.dense4 = Dense(in_channels=64, growth_rate=10, kernel_size=3, iteration=8, Nh=4, aug=True, shape = input_shape//8)
        ############################### 14
        self.transition4 = Transition(in_channels=80 + 64, out_channels=64)
        self.dense5 = Dense(in_channels=64, growth_rate=10, kernel_size=3, iteration=10, Nh=4, aug=True, shape = input_shape//16)
        ############################### 7
        self.transition5 = Transition(in_channels=100 + 64, out_channels=64)
        self.dense6 = Dense(in_channels=64, growth_rate=10, kernel_size=3, iteration=12, Nh=4, aug=True, shape = 7)
        ############################### 4
        self.transition6 = Transition(in_channels=120 + 64, out_channels=128)
        self.dense7 = Dense(in_channels=128, growth_rate=10, kernel_size=3, iteration=14, Nh=4, aug=True, shape = 4)
        ############################### 2
        self.transition7 = Transition(in_channels=140 + 128, out_channels=128)
        self.dense8 = Dense(in_channels=128, growth_rate=10, kernel_size=2, iteration=32, Nh=4, aug=True, shape = 2)
        self.aug_block = aug_block(in_channels = 320 + 128, out_channels = 100, kernel_size = 2, dk = 0.1,dv = 0.1, Nh = 10, shape = 2)
        self.avg_pool = nn.AvgPool2d(kernel_size = 2, stride = 2)
        self.conv = nn.Conv2d(in_channels = 100, out_channels = 42, kernel_size=1)
        self.relu = nn.ReLU()


    def forward(self, inputs):
        x = self.dense1(inputs)
        x = self.transition1(x)
        x = self.dense2(x)
        x = self.transition2(x)
        x = self.dense3(x)
        x = self.transition3(x)
        x = self.dense4(x)
        x = self.transition4(x)
        x = self.dense5(x)
        x = self.transition5(x)
        x = self.dense6(x)
        x = self.transition6(x)
        x = self.dense7(x)
        x = self.transition7(x)
        x = self.dense8(x)
        x = self.aug_block(x)
        x = self.avg_pool(x)
        x = self.conv(x)
        x = self.relu(x)
        x[x > 1.] = 1.
        x = torch.reshape(x, (-1, 21, 2))
        return x
if __name__ == "__main__":
    model = light_Model().cuda()
    x = torch.randn(1, 3, 224, 224).cuda()
    # print(model)
    # print(x)
    y = model(x)
    # y=aug_block(in_channels = 320 + 128, out_channels = 100, kernel_size = 2, dk = 0.1,dv = 0.1, Nh = 10, shape = 2)(x)
    print('y', y.shape, y.device)