import torch
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights
from model import get_basic_conv_layers
from torch.autograd import Variable

class CMAL(nn.Module):
    def __init__(self, num_class):
        super().__init__()
        net = resnet50(ResNet50_Weights.DEFAULT)
        net_layers = list(net.children())
        self.fe_0 = nn.Sequential(*net_layers[:5])
        self.fe_1 = nn.Sequential(*net_layers[5])
        self.fe_2 = nn.Sequential(*net_layers[6])
        self.fe_3 = nn.Sequential(*net_layers[7])

        self.mp1 = nn.MaxPool2d(kernel_size=56, stride=1)
        self.mp2 = nn.MaxPool2d(kernel_size=28, stride=1)
        self.mp3 = nn.MaxPool2d(kernel_size=14, stride=1)

        self.cb1 = nn.Sequential(
            *get_basic_conv_layers(512, 512, kernel_size=1, stride=1, padding=0, relu=True),
            *get_basic_conv_layers(512, 1024, kernel_size=3, stride=1, padding=1, relu=True)
        )

        self.c1 = nn.Sequential(
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ELU(inplace=True),
            nn.Linear(512, num_class)
        )

        self.cb2 = nn.Sequential(
            *get_basic_conv_layers(1024, 512, kernel_size=1, stride=1, padding=0, relu=True),
            *get_basic_conv_layers(512, 1024, kernel_size=3, stride=1, padding=1, relu=True)
        )
        self.c1 = nn.Sequential(
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ELU(inplace=True),
            nn.Linear(512, num_class),
        )

        self.cb3 = nn.Sequential(
            *get_basic_conv_layers(2048, 512, kernel_size=1, stride=1, padding=0, relu=True),
            *get_basic_conv_layers(512, 1024, kernel_size=3, stride=1, padding=1, relu=True)
        )
        self.c3 = nn.Sequential(
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ELU(inplace=True),
            nn.Linear(512, num_class),
        )

        self.c_all = nn.Sequential(
            nn.BatchNorm1d(1024 * 3),
            nn.Linear(1024 * 3, 512),
            nn.BatchNorm1d(512),
            nn.ELU(inplace=True),
            nn.Linear(512, num_class),
        )

    def forward(self, x):
        x = self.fe_0(x)
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



    