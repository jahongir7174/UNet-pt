import math

import torch
from torch.nn import functional


def add_weight_decay(model, weight_decay=1e-5):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    return [{'params': no_decay, 'weight_decay': 0.}, {'params': decay, 'weight_decay': weight_decay}]


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2.0 / (fan_out // m.groups)))
            if m.bias is not None:
                m.bias.data.zero_()
        if isinstance(m, torch.nn.BatchNorm2d):
            m.weight.data.fill_(1.0)
            m.bias.data.zero_()


class Conv(torch.nn.Module):
    def __init__(self, in_ch, out_ch, k=1, s=1, g=1):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_ch, out_ch, k, s, k // 2, groups=g, bias=False)
        self.norm = torch.nn.BatchNorm2d(out_ch)
        self.silu = torch.nn.SiLU(inplace=True)

    def forward(self, x):
        return self.silu(self.norm(self.conv(x)))


class SE(torch.nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.se = torch.nn.Sequential(torch.nn.Conv2d(ch, ch // 4, 1),
                                      torch.nn.SiLU(inplace=True),
                                      torch.nn.Conv2d(ch // 4, ch, 1),
                                      torch.nn.Sigmoid())

    def forward(self, x):
        return x * self.se(x.mean((2, 3), keepdim=True))


class Residual(torch.nn.Module):
    def __init__(self, ch, add):
        super().__init__()
        self.res = torch.nn.Sequential(Conv(ch, ch, 1),
                                       SE(ch),
                                       Conv(ch, ch, 3))
        self.add = add

    def forward(self, x):
        return x + self.res(x) if self.add else self.res(x)


class CSP(torch.nn.Module):
    def __init__(self, in_ch, out_ch, n=1, add=True):
        super().__init__()
        self.conv1 = Conv(in_ch, out_ch // 2)
        self.conv2 = Conv(in_ch, out_ch // 2)
        self.conv3 = Conv(out_ch, out_ch)
        self.res_m = torch.nn.Sequential(*[Residual(out_ch // 2, add) for _ in range(n)])

    def forward(self, x):
        return self.conv3(torch.cat((self.conv1(x), self.res_m(self.conv2(x))), dim=1))


class ASPPConv(torch.nn.Module):
    def __init__(self, in_ch, out_ch, d):
        super().__init__()
        self.res = torch.nn.Sequential(torch.nn.Conv2d(in_ch, out_ch, 3, 1, d, d, out_ch, False),
                                       torch.nn.BatchNorm2d(out_ch),
                                       torch.nn.SiLU())

    def forward(self, x):
        return self.res(x)


class ASPPPooling(torch.nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = Conv(in_ch, out_ch)

    def forward(self, x):
        size = x.shape[-2:]
        x = self.conv(x.mean((2, 3), keepdim=True))
        return functional.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPPModule(torch.nn.Module):
    def __init__(self, in_ch, out_ch, rates=(6, 12, 18)):
        super().__init__()
        self.module1 = Conv(in_ch, out_ch)

        self.module2 = ASPPConv(in_ch, out_ch, rates[0])
        self.module3 = ASPPConv(in_ch, out_ch, rates[1])
        self.module4 = ASPPConv(in_ch, out_ch, rates[2])

        self.module5 = ASPPPooling(in_ch, out_ch)

        self.project = torch.nn.Sequential(Conv(5 * out_ch, out_ch),
                                           torch.nn.Dropout(p=0.5))

    def forward(self, x):
        res = torch.cat((self.module1(x),
                         self.module2(x),
                         self.module3(x),
                         self.module4(x),
                         self.module5(x)), dim=1)
        return self.project(res)


class UNet(torch.nn.Module):
    def __init__(self, num_class):
        super().__init__()
        num_dep = [4, 12]
        filters = [3, 64, 128, 256, 512, 1024]
        self.b0 = Conv(filters[0], filters[1], 3, 1)
        self.b1 = Conv(filters[1], filters[2], 3, 2)
        self.b2 = CSP(filters[2], filters[2], num_dep[0])
        self.b3 = Conv(filters[2], filters[3], 3, 2)
        self.b4 = CSP(filters[3], filters[3], num_dep[1])
        self.b5 = Conv(filters[3], filters[4], 3, 2)
        self.b6 = CSP(filters[4], filters[4], num_dep[1])
        self.b7 = Conv(filters[4], filters[5], 3, 2)
        self.b8 = CSP(filters[5], filters[5], num_dep[0], False)
        self.b9 = ASPPModule(filters[5], filters[5])

        self.up = torch.nn.Upsample(None, 2)

        self.h0 = Conv(filters[5], filters[4], 3, 1)
        self.h1 = CSP(filters[5], filters[4], num_dep[0], False)
        self.h2 = Conv(filters[4], filters[3], 3, 1)
        self.h3 = CSP(filters[4], filters[3], num_dep[0], False)
        self.h4 = Conv(filters[3], filters[2], 3, 1)
        self.h5 = CSP(filters[3], filters[2], num_dep[0], False)
        self.h6 = Conv(filters[2], filters[1], 3, 1)
        self.h7 = CSP(filters[2], filters[1], num_dep[0], False)
        self.h8 = torch.nn.Sequential(Conv(filters[1], filters[1], 3),
                                      torch.nn.Conv2d(filters[1], num_class, 1))
        initialize_weights(self)

    def forward(self, x):
        b0 = self.b0(x)
        b1 = self.b1(b0)
        b2 = self.b2(b1)

        b3 = self.b3(b2)
        b4 = self.b4(b3)

        b5 = self.b5(b4)
        b6 = self.b6(b5)

        b7 = self.b7(b6)
        b8 = self.b8(b7)
        b9 = self.b9(b8)

        h0 = self.h0(b9)
        h1 = self.h1(torch.cat([self.up(h0), b6], 1))

        h2 = self.h2(h1)
        h3 = self.h3(torch.cat([self.up(h2), b4], 1))

        h4 = self.h4(h3)
        h5 = self.h5(torch.cat([self.up(h4), b2], 1))

        h6 = self.h6(h5)
        h7 = self.h7(torch.cat([self.up(h6), b0], 1))

        return self.h8(h7)


class StepLR:
    def __init__(self, optimizer):
        self.optimizer = optimizer

        for param_group in self.optimizer.param_groups:
            param_group.setdefault('initial_lr', param_group['lr'])

        self.base_values = [param_group['initial_lr'] for param_group in self.optimizer.param_groups]
        self.update_groups(self.base_values)

        self.decay_rate = 0.98
        self.decay_epochs = 5
        self.warmup_epochs = 8
        self.warmup_lr_init = 1e-6

        self.warmup_steps = [(v - self.warmup_lr_init) / self.warmup_epochs for v in self.base_values]
        self.update_groups(self.warmup_lr_init)

    def __str__(self) -> str:
        return 'step'

    def step(self, epoch: int) -> None:
        if epoch < self.warmup_epochs:
            values = [self.warmup_lr_init + epoch * s for s in self.warmup_steps]
        else:
            values = [v * (self.decay_rate ** (epoch // self.decay_epochs)) for v in self.base_values]
        if values is not None:
            self.update_groups(values)

    def update_groups(self, values):
        if not isinstance(values, (list, tuple)):
            values = [values] * len(self.optimizer.param_groups)
        for param_group, value in zip(self.optimizer.param_groups, values):
            param_group['lr'] = value


class RMSprop(torch.optim.Optimizer):
    """
    [https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf]
    """

    def __init__(self, params, lr=1e-2, alpha=0.9, eps=1e-10, weight_decay=0, momentum=0.,
                 centered=False, decoupled_decay=False, lr_in_momentum=True):

        defaults = dict(lr=lr, alpha=alpha, eps=eps, weight_decay=weight_decay, momentum=momentum,
                        centered=centered, decoupled_decay=decoupled_decay, lr_in_momentum=lr_in_momentum)
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for param_group in self.param_groups:
            param_group.setdefault('momentum', 0)
            param_group.setdefault('centered', False)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for param_group in self.param_groups:
            for param in param_group['params']:
                if param.grad is None:
                    continue
                grad = param.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Optimizer does not support sparse gradients')
                state = self.state[param]
                if len(state) == 0:
                    state['step'] = 0
                    state['square_avg'] = torch.ones_like(param.data)
                    if param_group['momentum'] > 0:
                        state['momentum_buffer'] = torch.zeros_like(param.data)
                    if param_group['centered']:
                        state['grad_avg'] = torch.zeros_like(param.data)

                square_avg = state['square_avg']
                one_minus_alpha = 1. - param_group['alpha']

                state['step'] += 1

                if param_group['weight_decay'] != 0:
                    if 'decoupled_decay' in param_group and param_group['decoupled_decay']:
                        param.data.add_(param.data, alpha=-param_group['weight_decay'])
                    else:
                        grad = grad.add(param.data, alpha=param_group['weight_decay'])

                square_avg.add_(grad.pow(2) - square_avg, alpha=one_minus_alpha)

                if param_group['centered']:
                    grad_avg = state['grad_avg']
                    grad_avg.add_(grad - grad_avg, alpha=one_minus_alpha)
                    avg = square_avg.addcmul(grad_avg, grad_avg, value=-1).add(param_group['eps']).sqrt_()
                else:
                    avg = square_avg.add(param_group['eps']).sqrt_()

                if param_group['momentum'] > 0:
                    buf = state['momentum_buffer']
                    if 'lr_in_momentum' in param_group and param_group['lr_in_momentum']:
                        buf.mul_(param_group['momentum']).addcdiv_(grad, avg, value=param_group['lr'])
                        param.data.add_(-buf)
                    else:
                        buf.mul_(param_group['momentum']).addcdiv_(grad, avg)
                        param.data.add_(-param_group['lr'], buf)
                else:
                    param.data.addcdiv_(grad, avg, value=-param_group['lr'])

        return loss
