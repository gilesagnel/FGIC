import torch
from torch import nn
from torchvision.models import resnext50_32x4d, resnet50,resnet34, ResNet50_Weights, ResNet34_Weights, ResNeXt50_32X4D_Weights

class CMAL(nn.Module):
    def __init__(self, num_class, ks, base_model):
        super().__init__()
        if base_model == "resnet34":
            net = resnet34(weights=ResNet34_Weights.DEFAULT)
            nf = 128
        elif base_model == "resnet50":
            net = resnet50(weights=ResNet50_Weights.DEFAULT)
            nf = 512
        else:
            net = resnext50_32x4d(weights=ResNeXt50_32X4D_Weights.DEFAULT)
            nf = 512

        net_layers = list(net.children())
        self.fe_1 = nn.Sequential(*net_layers[:6])
        self.fe_2 = nn.Sequential(*net_layers[6])
        self.fe_3 = nn.Sequential(*net_layers[7])


        self.mp1 = nn.MaxPool2d(kernel_size=ks, stride=1)
        self.mp2 = nn.MaxPool2d(kernel_size=ks//2, stride=1)
        self.mp3 = nn.MaxPool2d(kernel_size=ks//4, stride=1)

        self.cb1 = nn.Sequential(
            *get_basic_conv_layers(nf, nf, kernel_size=1, stride=1, padding=0, relu=True),
            *get_basic_conv_layers(nf, nf * 2, kernel_size=3, stride=1, padding=1, relu=True)
        )

        self.c1 = nn.Sequential(
            nn.BatchNorm1d(nf * 2),
            nn.Linear(nf * 2, nf),
            nn.BatchNorm1d(nf),
            nn.ELU(inplace=True),
            nn.Linear(nf, num_class)
        )

        self.cb2 = nn.Sequential(
            *get_basic_conv_layers(nf * 2, nf, kernel_size=1, stride=1, padding=0, relu=True),
            *get_basic_conv_layers(nf, nf * 2, kernel_size=3, stride=1, padding=1, relu=True)
        )
        self.c2 = nn.Sequential(
            nn.BatchNorm1d(nf * 2),
            nn.Linear(nf * 2, nf),
            nn.BatchNorm1d(nf),
            nn.ELU(inplace=True),
            nn.Linear(nf, num_class),
        )

        self.cb3 = nn.Sequential(
            *get_basic_conv_layers(nf * 4, nf, kernel_size=1, stride=1, padding=0, relu=True),
            *get_basic_conv_layers(nf, nf * 2, kernel_size=3, stride=1, padding=1, relu=True)
        )
        self.c3 = nn.Sequential(
            nn.BatchNorm1d(nf * 2),
            nn.Linear(nf * 2, nf),
            nn.BatchNorm1d(nf),
            nn.ELU(inplace=True),
            nn.Linear(nf, num_class),
        )

        self.c_all = nn.Sequential(
            nn.BatchNorm1d(nf * 6),
            nn.Linear(nf * 6, nf),
            nn.BatchNorm1d(nf),
            nn.ELU(inplace=True),
            nn.Linear(nf, num_class),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for name, param in self.named_parameters():
            if 'fe_1' in name or 'fe_2' in name or 'fe_3' in name:
                continue  
            
            if len(param.shape) < 2:
                continue
            
            if 'weight' in name:
                nn.init.kaiming_normal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, x):
        x1 = self.fe_1(x)
        x2 = self.fe_2(x1)
        x3 = self.fe_3(x2)
        
        x1_ = self.cb1(x1)
        map1 = x1_.detach()
        x1_ = self.mp1(x1_)
        x1_f = x1_.view(x1_.size(0), -1)

        x1_c = self.c1(x1_f)

        x2_ = self.cb2(x2)
        map2 = x2_.detach()
        x2_ = self.mp2(x2_)
        x2_f = x2_.view(x2_.size(0), -1)
        x2_c = self.c2(x2_f)

        x3_ = self.cb3(x3)
        map3 = x3_.detach()
        x3_ = self.mp3(x3_)
        x3_f = x3_.view(x3_.size(0), -1)
        x3_c = self.c3(x3_f)

        x_c_all = torch.cat((x1_f, x2_f, x3_f), -1)
        x_c_all = self.c_all(x_c_all)

        return x1_c, x2_c, x3_c, x_c_all, map1, map2, map3

def get_basic_conv_layers(in_nc, out_nc, kernel_size, stride=1, padding=0, 
                        dilation=1, groups=1, relu=True, bn=True, bias=False):
    layers = [nn.Conv2d(in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=padding, 
                            dilation=dilation, groups=groups, bias=bias)]
    if bn:
        layers += [nn.BatchNorm2d(out_nc, eps=1e-5,
                                momentum=0.01, affine=True)]
    if relu:
        layers += [nn.ReLU()]

    return layers
