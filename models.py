import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import numpy as np
from layer_defs import *


def get_convlayer(in_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False, conv_type="Plain"):
    if(conv_type=="Plain"):
        l = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, bias=bias)
    elif(conv_type=="sWS"):
        l = ScaledStdConv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, bias=bias)
    elif(conv_type=="WeightNormalized"):
        l = WN_self(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, bias=bias)
    elif(conv_type=="WeightCentered"):
        l = WCConv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, bias=bias)

    torch.nn.init.kaiming_normal_(l.weight, mode='fan_in', nonlinearity='relu')
    try:
        torch.nn.init.zeros_(l.bias)
    except:
        pass
    return l

def get_norm_and_activ_layer(norm_type, num_channels, n_groups):
    if(norm_type=="Plain"):
        l = [nn.Identity(), nn.ReLU(inplace=True)]
    elif(norm_type=="BatchNorm"):
        l = [BN_self(num_channels), nn.ReLU(inplace=True)]
    elif(norm_type=="LayerNorm"):
        l = [LN_self(num_channels), nn.ReLU(inplace=True)]
    elif(norm_type=="Instance Normalization"):
        l = [IN_self(num_channels), nn.ReLU(inplace=True)]
    elif(norm_type=="GroupNorm"):
        l = [GN_self(num_channels, groups=n_groups), nn.ReLU(inplace=True)]
    elif(norm_type=="Filter Response Normalization"):
        l = [FRN_self(num_channels), TLU(num_channels)]
    elif(norm_type=="Weight Normalization"):
        l = [nn.Identity(), WN_scaledReLU(inplace=True)]
    elif(norm_type=="Scaled Weight Standardization"):
        l = [nn.Identity(), nn.ReLU(inplace=True)]
    elif(norm_type=="EvoNormBO"):
        l = [nn.Identity(), EvoNormBO(num_features=num_channels)]
    elif(norm_type=="EvoNormSO"):
        l = [nn.Identity(), EvoNormSO(num_features=num_channels)]
    elif(norm_type=="Variance Normalization"):
        l = [VN_self(num_channels), nn.ReLU(inplace=True)]
    elif(norm_type=="Mean Centering"):
        l = [MC_self(num_channels), nn.ReLU(inplace=True)]
    return l

def get_model(arch, cfg_use, conv_type="Plain", norm_type="BatchNorm", p_grouping=1, n_classes=100, probe=True, skipinit=False, preact=False, group_list=None):
    conv_type = "WeightNormalized" if(norm_type=="Weight Normalization") else "sWS" if(norm_type=="Scaled Weight Standardization") else conv_type 

    if(arch=="VGG"):
        model = VGG(cfg_use, conv_type=conv_type, norm_type=norm_type, p_grouping=p_grouping, n_classes=n_classes, probe=probe, group_list=group_list)
    elif(arch=="ResNet-56"):
        model = ResNet56(conv_type=conv_type, norm_type=norm_type, p_grouping=p_grouping, n_classes=n_classes, probe=probe, skipinit=skipinit, preact=preact)
    return model


class VGG(nn.Module):
    def __init__(self, cfg, conv_type="Plain", norm_type="BatchNorm", n_classes=100, conv_bias=False, p_grouping=1.0, probe=True, group_list=None):
        super(VGG, self).__init__()
        self.conv_type = conv_type
        self.norm_type = norm_type
        self.probe = probe
        self.conv_bias = (norm_type=="Scaled Weight Standardization") or (norm_type=="Weight Normalization")
        self.p_grouping = p_grouping
        self.group_list = group_list
        
        self.features = self._make_layers(cfg)
        self.classifier = nn.Linear(cfg[-1], n_classes)

        if self.probe:
            self.params_list = []
            self.grads_list = []
            for _ in cfg:
                self.params_list.append([])
                self.grads_list.append([])

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for gid, x in enumerate(cfg):
            if type(x) == tuple:
                n_groups = (int(self.p_grouping) if (self.p_grouping>1) else int(np.ceil(x[0] * self.p_grouping))) if self.group_list==None else self.group_list[gid]
                layers += [get_convlayer(conv_type=self.conv_type, in_channels=in_channels, out_channels=x[0], kernel_size=3, padding=1, stride=2, bias=self.conv_bias),
                           Conv_prober() if self.probe else nn.Identity()]
                layers += get_norm_and_activ_layer(norm_type=self.norm_type, num_channels=x[0], n_groups=n_groups)
                layers += [Activs_prober() if self.probe else nn.Identity()]
                in_channels = x[0]
            else:
                n_groups = (int(self.p_grouping) if (self.p_grouping>1) else int(np.ceil(x * self.p_grouping))) if self.group_list==None else self.group_list[gid]
                layers += [get_convlayer(conv_type=self.conv_type, in_channels=in_channels, out_channels=x, kernel_size=3, padding=1, bias=self.conv_bias),
                           Conv_prober() if self.probe else nn.Identity()]
                layers += get_norm_and_activ_layer(norm_type=self.norm_type, num_channels=x, n_groups=n_groups)
                layers += [Activs_prober() if self.probe else nn.Identity()]
                in_channels = x
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.features(x)
        out = out.mean(dim=(2,3))
        out = self.classifier(out)
        return out


######### ResNet-56 #########
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, conv_type='Plain', norm_type='BatchNorm', p_grouping=1.0, probe=True, bias=False, skipinit=False):
        super(BasicBlock, self).__init__()
        self.skip_gain = nn.Parameter(torch.Tensor([0])) if skipinit else 1

        ### Layer 1
        n_groups = int(p_grouping) if (p_grouping>1) else int(np.ceil(planes * p_grouping))
        self.conv1 = get_convlayer(conv_type=conv_type, in_channels=in_planes, out_channels=planes, kernel_size=3, padding=1, stride=stride, bias=bias)
        self.conv_probe1 = Conv_prober() if probe else nn.Identity()
        
        norm_and_activ = get_norm_and_activ_layer(norm_type=norm_type, num_channels=planes, n_groups=n_groups)
        self.norm1 = norm_and_activ[0]
        self.activ1 = norm_and_activ[1] 
        self.norm_probe1 = Activs_prober() if probe else nn.Identity()

        ### Layer 2
        self.conv2 = get_convlayer(conv_type=conv_type, in_channels=planes, out_channels=planes, kernel_size=3, padding=1, stride=1, bias=bias)
        self.conv_probe2 = Conv_prober() if probe else nn.Identity()

        norm_and_activ = get_norm_and_activ_layer(norm_type=norm_type, num_channels=planes, n_groups=n_groups)
        self.norm2 = norm_and_activ[0]
        self.activ2 = norm_and_activ[1] 
        self.norm_probe2 = Activs_prober() if probe else nn.Identity()

        ### Shortcut
        self.shortcut = nn.Sequential()
        norm_and_activ = get_norm_and_activ_layer(norm_type=norm_type, num_channels=self.expansion*planes, n_groups=n_groups)

        if stride != 1 or in_planes != self.expansion*planes:
            n_groups = int(p_grouping) if (p_grouping>1) else int(np.ceil(self.expansion*planes * p_grouping))
            self.shortcut = nn.Sequential(
                get_convlayer(conv_type=conv_type, in_channels=in_planes, out_channels=self.expansion*planes, kernel_size=3, padding=1, stride=stride, bias=bias),
                Conv_prober() if probe else nn.Identity(),
                norm_and_activ[0],
            )

    def forward(self, x):
        out = self.conv_probe1(self.conv1(x))
        out = self.norm_probe1(self.activ1(self.norm1(out)))
        out = self.conv_probe2(self.conv2(out))
        
        # Standard ResNet
        out = self.skip_gain * self.norm2(out)
        out += self.shortcut(x)
        out = self.norm_probe2(self.activ2(out))

        # # Activation before residual ResNet (Used in WeightNorm experiments)
        # out = self.skip_gain * self.activ2(self.norm2(out))
        # out += self.shortcut(x)
        # out = self.norm_probe2(out)
        return out

class preact_BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, conv_type='Plain', norm_type='BatchNorm', p_grouping=1.0, probe=True, bias=False, skipinit=False):
        super(preact_BasicBlock, self).__init__()
        self.skip_gain = nn.Parameter(torch.Tensor([0])) if skipinit else 1

        ### Layer 1
        n_groups = int(p_grouping) if (p_grouping>1) else int(np.ceil(planes * p_grouping))

        norm_and_activ = get_norm_and_activ_layer(norm_type=norm_type, num_channels=in_planes, n_groups=n_groups)
        self.norm1 = norm_and_activ[0]
        self.activ1 = norm_and_activ[1] 
        self.norm_probe1 = Activs_prober() if probe else nn.Identity()

        self.conv1 = get_convlayer(conv_type=conv_type, in_channels=in_planes, out_channels=planes, kernel_size=3, padding=1, stride=stride, bias=bias)
        self.conv_probe1 = Conv_prober() if probe else nn.Identity()
        
        ### Layer 2
        norm_and_activ = get_norm_and_activ_layer(norm_type=norm_type, num_channels=planes, n_groups=n_groups)
        self.norm2 = norm_and_activ[0]
        self.activ2 = norm_and_activ[1] 
        self.norm_probe2 = Activs_prober() if probe else nn.Identity()

        self.conv2 = get_convlayer(conv_type=conv_type, in_channels=planes, out_channels=planes, kernel_size=3, padding=1, stride=1, bias=bias)
        self.conv_probe2 = Conv_prober() if probe else nn.Identity()

        ### Shortcut
        self.shortcut = nn.Sequential()

        if stride != 1 or in_planes != self.expansion*in_planes:
            n_groups = int(p_grouping) if (p_grouping>1) else int(np.ceil(self.expansion*in_planes * p_grouping))
            self.shortcut = nn.Sequential(
                get_norm_and_activ_layer(norm_type=norm_type, num_channels=self.expansion*in_planes, n_groups=n_groups)[0],
                get_convlayer(conv_type=conv_type, in_channels=in_planes, out_channels=self.expansion*planes, kernel_size=3, padding=1, stride=stride, bias=bias),
                Conv_prober() if probe else nn.Identity(),
            )

    def forward(self, x):
        out = self.norm_probe1(self.activ1(self.norm1(x)))
        out = self.conv_probe1(self.conv1(out))

        out = self.activ2(self.norm2(out))
        out = self.skip_gain * self.conv_probe2(self.conv2(out))

        out += (self.shortcut(x))
        return self.norm_probe2(out)

class ResNet_cifar(nn.Module):
    def __init__(self, block, num_blocks, conv_type='Plain', norm_type='BatchNorm', n_classes=100, p_grouping=1.0, probe=True, skipinit=False):
        super(ResNet_cifar, self).__init__()
        conv_bias = (norm_type=="Scaled Weight Standardization") or (norm_type=="Weight Normalization")

        self.probe = probe
        if self.probe:
            self.params_list = []
            self.grads_list = []
            for _ in range(57):
                self.params_list.append([])
                self.grads_list.append([])

        ### Opening Layer
        n_groups = int(p_grouping) if (p_grouping>1) else int(np.ceil(32 * p_grouping))
        self.conv1 = get_convlayer(conv_type=conv_type, in_channels=3, out_channels=32, kernel_size=3, padding=1, stride=1, bias=conv_bias)
        self.conv_probe1 = Conv_prober() if probe else nn.Identity()
        norm_and_activ = get_norm_and_activ_layer(norm_type=norm_type, num_channels=32, n_groups=n_groups)
        self.norm1 = norm_and_activ[0]
        self.activ1 = norm_and_activ[1] 
        self.norm_probe1 = Activs_prober() if probe else nn.Identity()

        ### Residual Layers
        self.in_planes = 32
        self.layer1 = self._make_layer(block, 32, num_blocks[0], stride=1, conv_type=conv_type, norm_type=norm_type, probe=probe, p_grouping=p_grouping, bias=conv_bias, skipinit=skipinit)
        self.layer2 = self._make_layer(block, 64, num_blocks[1], stride=2, conv_type=conv_type, norm_type=norm_type, probe=probe, p_grouping=p_grouping, bias=conv_bias, skipinit=skipinit)
        self.layer3 = self._make_layer(block, 128, num_blocks[2], stride=2, conv_type=conv_type, norm_type=norm_type, probe=probe, p_grouping=p_grouping, bias=conv_bias, skipinit=skipinit)
        self.linear = nn.Linear(128*block.expansion, n_classes)

    def _make_layer(self, block, planes, num_blocks, stride, conv_type="Plain", norm_type="BatchNorm", probe=True, p_grouping=1.0, bias=False, skipinit=False):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, conv_type=conv_type, norm_type=norm_type, probe=probe, p_grouping=p_grouping, bias=bias, skipinit=skipinit))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv_probe1(self.conv1(x))
        out = self.norm_probe1(self.activ1(self.norm1(out)))

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet56(conv_type='Plain', norm_type='BatchNorm', n_classes=100, p_grouping=1.0, probe=True, skipinit=False, preact=False):
    if(preact):
        return ResNet_cifar(preact_BasicBlock, [9,9,9], conv_type=conv_type, norm_type=norm_type, n_classes=n_classes, p_grouping=p_grouping, probe=probe, skipinit=skipinit)
    else:
        return ResNet_cifar(BasicBlock, [9,9,9], conv_type=conv_type, norm_type=norm_type, n_classes=n_classes, p_grouping=p_grouping, probe=probe, skipinit=skipinit)