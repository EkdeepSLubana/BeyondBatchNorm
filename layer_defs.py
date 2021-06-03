import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
pi = torch.acos(torch.zeros(1)).item() * 2

###############  Helper functions ###############
def instance_std(x, eps=1e-5):
    var = torch.var(x, dim = (2, 3), keepdim=True).expand_as(x)
    if torch.isnan(var).any():
        var = torch.zeros(var.shape).to(x.device)
    return torch.sqrt(var + eps)

def group_std(x, groups = 32, eps = 1e-5):
    N, C, H, W = x.size()
    x = torch.reshape(x, (N, groups, C // groups, H, W))
    var = torch.var(x, dim = (2, 3, 4), keepdim = True).expand_as(x)
    return torch.reshape(torch.sqrt(var + eps), (N, C, H, W))

def group_mean(x, groups=32):
    N, C, H, W = x.size()
    x = torch.reshape(x, (N, groups, C // groups, H, W))
    mean = torch.mean(x, dim = (2, 3, 4), keepdim = True).expand_as(x)
    return torch.reshape(mean, (N, C, H, W))

class WCConv2d(nn.Conv2d):
    def __init__(
            self, in_channels, out_channels, kernel_size, stride=1, padding=None, dilation=1, groups=1,
            bias=True, gamma=1.0, eps=1e-5):
        if padding is None:
            padding = get_padding(kernel_size, stride, dilation)
        super().__init__(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias)

    def get_weight(self):
        weight = self.weight - torch.mean(self.weight, dim=[1, 2, 3], keepdim=True)
        return weight

    def forward(self, x):
        return F.conv2d(x, self.get_weight(), self.bias, self.stride, self.padding, self.dilation, self.groups)


############### Parametric Normalization Layers ############### 
### Scaled Weight Standardization (Brock et al., 2021) ###
# (based on https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/nfnet.py) 
class ScaledStdConv2d(nn.Conv2d):
    def __init__(
            self, in_channels, out_channels, kernel_size, stride=1, padding=None, dilation=1, groups=1,
            bias=True, scale_activ=1.0, eps=1e-5):
        if padding is None:
            padding = get_padding(kernel_size, stride, dilation)
        super().__init__(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias)
        self.gamma = nn.Parameter(torch.ones(self.out_channels, 1, 1, 1))
        self.scale = scale_activ / (self.weight[0].numel() ** 0.5)  # gamma * 1 / sqrt(fan-in)
        self.eps = eps

    def get_weight(self):
        std, mean = torch.std_mean(self.weight, dim=[1, 2, 3], keepdim=True, unbiased=False)
        weight = (self.weight - mean) / (std + self.eps)
        return (2 * pi / (pi - 1)) ** 0.5 * self.gamma * self.scale * weight

    def forward(self, x):
        return F.conv2d(x, self.get_weight(), self.bias, self.stride, self.padding, self.dilation, self.groups)


### Weight Normalization (Salimans and Kingma, 2016) ###
class WN_self(nn.Conv2d):
    def __init__(
            self, in_channels, out_channels, kernel_size, stride=1, padding=None, dilation=1, groups=1, bias=True, eps=1e-5):
        if padding is None:
            padding = get_padding(kernel_size, stride, dilation)
        super().__init__(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias)
        self.gamma = nn.Parameter(torch.ones(self.out_channels, 1, 1, 1))
        self.eps = eps

    def get_weight(self):
        denom = torch.linalg.norm(self.weight, dim=[1, 2, 3], keepdim=True)
        weight = self.weight / (denom + self.eps)
        return self.gamma * weight

    def forward(self, x):
        return F.conv2d(x, self.get_weight(), self.bias, self.stride, self.padding, self.dilation, self.groups)

# Scaled activation function for WeightNorm (Performs scaled/bias correction; Arpit et al., 2016)
class WN_scaledReLU(nn.ReLU):
    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        return (2 * pi / (pi - 1))**0.5 * (F.relu(x, inplace=self.inplace) - (1 / (2 * pi))**(0.5))


############### Activations-based normalization layers ###############
### BatchNorm (Ioffe and Szegedy, 2015) ###
# (based on https://pytorch.org/docs/stable/_modules/torch/nn/modules/batchnorm.html#BatchNorm2d)
class BN_self(nn.Module):
    def __init__(self, num_features, momentum=0.9, eps=1e-5):
        super().__init__()
        shape = (1, num_features, 1, 1)
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        self.momentum = momentum
        self.eps = eps

        self.register_buffer('moving_mean', torch.ones(shape))
        self.register_buffer('moving_var', torch.ones(shape))
        self.reset_parameters()

    def reset_parameters(self):
        self.moving_var.fill_(1)

    def forward(self, X):
        if self.training:
            var, mean = torch.var_mean(X, dim=(0, 2, 3), keepdim=True, unbiased=False)
            self.moving_mean.mul_(self.momentum)
            self.moving_mean.add_((1 - self.momentum) * mean)
            self.moving_var.mul_(self.momentum)
            self.moving_var.add_((1 - self.momentum) * var)
        else:
            var = self.moving_var
            mean = self.moving_mean

        X = (X - mean) * torch.rsqrt(var+self.eps)
        return X * self.gamma + self.beta


### LayerNorm (Ba et al., 2016) ###
class LN_self(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        shape = (1, num_features, 1, 1)
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))

    def forward(self, X, eps=1e-5):
        var, mean = torch.var_mean(X, dim=(1, 2, 3), keepdim=True, unbiased=False)
        X = (X - mean) / torch.sqrt(var + eps) 
        return self.gamma * X + self.beta  


### InstanceNorm (Ulyanov et al., 2017) ###
class IN_self(nn.Module):
    def __init__(self, num_features):
        super(IN_self, self).__init__()
        self.num_features = num_features
        self.gamma = nn.Parameter(torch.Tensor(1, num_features, 1, 1), requires_grad=True)
        self.beta = nn.Parameter(torch.Tensor(1, num_features, 1, 1), requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.gamma)
        nn.init.zeros_(self.beta)

    def forward(self, X, eps=1e-5):
        var, mean = torch.var_mean(X, dim=(2, 3), keepdim=True, unbiased=False)
        X = (X - mean) / torch.sqrt(var + eps) 
        return self.gamma * X + self.beta


### GroupNorm (Wu and He, 2018) ###
class GN_self(nn.Module):
    def __init__(self, num_features, groups=32):
        super(GN_self, self).__init__()
        self.num_features = num_features
        self.groups = groups
        self.gamma = nn.Parameter(torch.Tensor(1, num_features, 1, 1), requires_grad=True)
        self.beta = nn.Parameter(torch.Tensor(1, num_features, 1, 1), requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.gamma)
        nn.init.zeros_(self.beta)

    def forward(self, x, eps=1e-5):
        me2 = group_mean(x, groups=self.groups)
        nu2 = group_std(x, groups=self.groups, eps=eps)
        x = (x-me2) / (nu2)
        return self.gamma * x + self.beta


### Filter Response Normalization (Singh and Krishnan, 2019) ###
class TLU(nn.Module):
    def __init__(self, num_features):
        """max(y, tau) = max(y - tau, 0) + tau = ReLU(y - tau) + tau"""
        super(TLU, self).__init__()
        self.num_features = num_features
        self.tau = nn.Parameter(torch.Tensor(1, num_features, 1, 1), requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.tau)

    def extra_repr(self):
        return 'num_features={num_features}'.format(**self.__dict__)

    def forward(self, x):
        return torch.max(x, self.tau)


class FRN_self(nn.Module):
    def __init__(self, num_features, eps=1e-5, is_eps_learnable=True):
        super(FRN_self, self).__init__()

        self.num_features = num_features
        self.init_eps = eps
        self.is_eps_learnable = is_eps_learnable

        self.gamma = nn.Parameter(torch.Tensor(1, num_features, 1, 1), requires_grad=True)
        self.beta = nn.Parameter(torch.Tensor(1, num_features, 1, 1), requires_grad=True)
        self.eps = nn.Parameter(torch.Tensor(1), requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.gamma)
        nn.init.zeros_(self.beta)
        if self.is_eps_learnable:
            nn.init.constant_(self.eps, self.init_eps)

    def extra_repr(self):
        return 'num_features={num_features}, eps={init_eps}'.format(**self.__dict__)

    def forward(self, x):
        nu2 = x.pow(2).mean(dim=[2, 3], keepdim=True)
        x = x * torch.rsqrt(nu2 + self.eps.abs())
        return self.gamma * x + self.beta


### Variance Normalization (Daneshmand et al., 2020) ###
# Essentially an ablation of BatchNorm that the authors found to be as successful as BatchNorm
class VN_self(nn.Module):
    def __init__(self, num_features, momentum=0.9, eps=1e-5):
        super().__init__()
        shape = (1, num_features, 1, 1)
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        self.momentum = momentum
        self.eps = eps

        self.register_buffer('moving_var', torch.ones(shape))
        self.reset_parameters()

    def reset_parameters(self):
        self.moving_var.fill_(1)

    def forward(self, X):
        if self.training:
            var = torch.var(X, dim=(0, 2, 3), keepdim=True, unbiased=False)
            self.moving_var.mul_(self.momentum)
            self.moving_var.add_((1 - self.momentum) * var)
        else:
            var = self.moving_var

        X = X * torch.rsqrt(var+self.eps)
        return X * self.gamma + self.beta


############### AutoML designed layers (based on https://github.com/digantamisra98/EvoNorm) ###############
### EvoNormSO (Liu et al., 2020) ###
class EvoNormSO(nn.Module):
    def __init__(self, num_features, eps = 1e-5, groups = 32):
        super(EvoNormSO, self).__init__()
        self.groups = groups
        self.eps = eps
        self.num_features = num_features

        self.gamma = nn.Parameter(torch.ones(1, self.num_features, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, self.num_features, 1, 1))
        self.v = nn.Parameter(torch.ones(1,self.num_features, 1, 1))

    def forward(self, x):
        num = x * torch.sigmoid(self.v * x)   
        return num / group_std(x, groups = self.groups, eps = self.eps) * self.gamma + self.beta

### EvoNormBO (Liu et al., 2020) ###
class EvoNormBO(nn.Module):
    def __init__(self, num_features, momentum = 0.9, eps = 1e-5, training = True):
        super(EvoNormBO, self).__init__()
        self.training = training
        self.momentum = momentum
        self.eps = eps
        self.num_features = num_features

        self.gamma = nn.Parameter(torch.ones(1, self.num_features, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, self.num_features, 1, 1))
        self.v = nn.Parameter(torch.ones(1,self.num_features, 1, 1))
        self.register_buffer('moving_var', torch.ones(1, self.num_features, 1, 1))

        self.reset_parameters()

    def reset_parameters(self):
        self.moving_var.fill_(1)

    def forward(self, x):
        if self.moving_var.device != x.device:
            self.moving_var = self.moving_var.to(x.device)
        if self.training:
            var = torch.var(x, dim = (0, 2, 3), unbiased = False, keepdim = True)
            self.moving_var.mul_(self.momentum)
            self.moving_var.add_((1 - self.momentum) * var)
        else:
            var = self.moving_var

        den = torch.max((var+self.eps).sqrt(), self.v * x + instance_std(x, eps = self.eps))
        return x / den * self.gamma + self.beta


############### Prober layers for tracking statistics ###############
### Conv_prober (i.e., probes activations and gradients from convolutional layers) ###
class Conv_prober(nn.Module):
    def __init__(self):
        super(Conv_prober, self).__init__()
        self.std_list = []
        # Grads
        self.grads_norms = []

        class sim_grads(torch.autograd.Function):
            @staticmethod
            def forward(ctx, input):
                self.std_list.append(input.std(dim=[0,2,3]).mean().item())
                return input.clone()

            @staticmethod
            def backward(ctx, grad_output):
                M = grad_output.view(grad_output.shape[0], -1)
                # Gradient Norms
                self.grads_norms.append(M.norm().item())
                M = (M / (torch.linalg.norm(M, dim=[1], keepdim=True)+1e-10))
                M = torch.matmul(M, M.T)
                return grad_output.clone()
            
        self.cal_prop = sim_grads.apply

    def forward(self, input):
        if not torch.is_grad_enabled():
            return input
        else:
            return self.cal_prop(input)


### Activs_prober (i.e., probes activations) ###
class Activs_prober(nn.Module):
    def __init__(self):
        super(Activs_prober, self).__init__()
        # Activs
        self.activs_norms = []
        self.activs_corr = []
        self.activs_ranks = []

        class sim_activs(torch.autograd.Function):
            @staticmethod
            def forward(ctx, input):
                M = input.clone()
                # Activation Variance
                avar = M.var(dim=[0,2,3], keepdim=True)
                self.activs_norms.append(avar.mean().item())
                anorms = torch.linalg.norm(M, dim=[1,2,3], keepdim=True)
                # self.activs_norms.append(anorms.mean().item())
                M = (M / anorms).reshape(M.shape[0], -1)
                M = torch.matmul(M, M.T) 
                # Activation Correlations
                self.activs_corr.append(((M.sum(dim=1) - 1) / (M.shape[0]-1)).mean().item())
                # Activation Ranks (calculates stable rank)
                tr = torch.diag(M).sum()
                opnom = torch.linalg.norm(M, ord=2)
                self.activs_ranks.append((tr / opnom).item())
                return input.clone()

            @staticmethod
            def backward(ctx, grad_output):
                return grad_output.clone()
            
        self.cal_prop = sim_activs.apply

    def forward(self, input):
        if not torch.is_grad_enabled():
            return input
        else:
            return self.cal_prop(input)