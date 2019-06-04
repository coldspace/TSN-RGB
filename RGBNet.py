# -*- coding: utf-8 -*-
# @Time    : 2019/5/24 0:55
# @Author  : zwenc
# @File    : RGBNet.py

import torch        # torch 框架
import torchvision  # 有现成的数据集、网络，这里使用到其现成的网络
from torch import nn
import model_zoo

class RGBNet(nn.Module):
    def __init__(self,num_class,segments = 3, module="RGB",dropout=0):
        super(RGBNet, self).__init__()
        self.num_class = num_class              # 有多少个类
        self.num_segments = segments            # 视频分为3段
        self.dropout = dropout                  # 是否添加dropout层，默认不添加
        self.module = module
        self._enable_pbn = False

        if module == "RGB":                     # 加判断是为了后期拓展功能
            self.in_channels = 3                # RGB 输入是3个通道

        self._load_netmodule("resnet101")       # 加载resnet101网络模型

    def forward(self, input):
        """
        Args:
            input: 数据，只支持四维输入[batch, channel , weight, height]
        Returns: 网络训练后结果
        """
        base_out = self.base_module(input)

        if self.dropout > 0:
            base_out = self.new_fc(base_out)

        # base_out = self.lastLayers(base_out)

        return base_out

    def _load_netmodule(self, module):
        """
        Args:
            module:  选择的模型，参数类型str
        """
        if module == "resnet101":
            # True 表示加载了预训练值
            self.base_module = torchvision.models.resnet101(True) # type: torchvision.models.resnet.ResNet
            self.input_size = 224                                 # 每一个网络对应的输入图片尺寸不同
        else:
            return None

        self._change_First_conv()  # 修改第一层卷积，输入维度
        self._change_last_layer()  # 修改最后一层的输出维度

    def _change_First_conv(self):
        # 准备替换掉网络的输入层
        # 获得模型网络的构成，如conv1卷积层结构等
        base_model = list(self.base_module.modules())
        # print(base_model)

        # 并不是所有的first_conv都在第1个，我这里是打印了print看出来的，只适用于resnet101网络
        first_conv = base_model[1]    # type: torch.nn.Conv2d
        # print(first_conv)

        # 创建新的一层网络，但是没有权重信息，用
        new_conv = torch.nn.Conv2d(self.in_channels, first_conv.out_channels,first_conv.kernel_size,first_conv.stride,first_conv.padding,
                                   first_conv.dilation,first_conv.groups,first_conv.bias,first_conv.padding_mode)

        # 准备加载权重信息，params就是原来的first_conv内的权重信息
        params = [x.clone() for x in first_conv.parameters()][0] # type: torch.Tensor
        kernel_size = params.size()
        # torch.Size([64, 3, 7, 7])

        # 这种语法仅仅支持torch.Size类型，torch.tensor类型都不支持
        new_kernel_size = kernel_size[:1] + (self.in_channels, ) + kernel_size[2:]
        # mean对维度进行平均，expand扩展维度，contiguous如果tensor在内存上连续则返回原tensor，其实没用
        new_kernels = params.data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()

        # 更新到新的卷积层上
        new_conv.weight.data = new_kernels

        # 将新的卷积层，加载到网络上。并不是所有的第一层网络都叫conv1
        self.base_module.conv1 = new_conv

    def _change_last_layer(self):
        base_model = list(self.base_module.modules())

        # 并不是所有的都在len(base_model)-1这个位置
        Linear = base_model[len(base_model)-1]    # type: torch.nn.Linear
        # Linear(in_features=2048, out_features=1000, bias=True)
        # print(first_conv)

        # 更新权重信息
        if self.dropout == 0:
            self.base_module.fc = torch.nn.Linear(Linear.in_features,self.num_class)
            self.new_fc = None
            self._init_weight(self.base_module.fc)
        else:
            self.base_module.fc = torch.nn.Dropout(p=self.dropout)
            self.new_fc = torch.nn.Linear(Linear.in_features,self.num_class)
            self._init_weight(self.new_fc)

        # self.lastLayers = torch.nn.Softmax()

    def _init_weight(self,Linear:torch.nn.Linear):
        std = 0.001
        torch.nn.init.normal(Linear.weight, 0, std)
        torch.nn.init.constant(Linear.bias, 0)

    def Load_Data(self, path):
        print("load_data")

    def Save_Data(self, path):
        print("Save_Data")

    def Get_Module(self):
        return self.base_module

    def get_optim_policies(self):
        first_conv_weight = []
        first_conv_bias = []
        normal_weight = []
        normal_bias = []
        bn = []

        conv_cnt = 0
        bn_cnt = 0
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Conv1d):
                ps = list(m.parameters())
                conv_cnt += 1
                if conv_cnt == 1:
                    first_conv_weight.append(ps[0])
                    if len(ps) == 2:
                        first_conv_bias.append(ps[1])
                else:
                    normal_weight.append(ps[0])
                    if len(ps) == 2:
                        normal_bias.append(ps[1])
            elif isinstance(m, torch.nn.Linear):
                ps = list(m.parameters())
                normal_weight.append(ps[0])
                if len(ps) == 2:
                    normal_bias.append(ps[1])

            elif isinstance(m, torch.nn.BatchNorm1d):
                bn.extend(list(m.parameters()))
            elif isinstance(m, torch.nn.BatchNorm2d):
                bn_cnt += 1
                # later BN's are frozen
                if not self._enable_pbn or bn_cnt == 1:
                    bn.extend(list(m.parameters()))
            elif len(m._modules) == 0:
                if len(list(m.parameters())) > 0:
                    raise ValueError("New atomic module type: {}. Need to give it a learning policy".format(type(m)))

        return [
            {'params': first_conv_weight, 'lr_mult': 5 if self.module == 'Flow' else 1, 'decay_mult': 1,
             'name': "first_conv_weight"},
            {'params': first_conv_bias, 'lr_mult': 10 if self.module == 'Flow' else 2, 'decay_mult': 0,
             'name': "first_conv_bias"},
            {'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "normal_weight"},
            {'params': normal_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "normal_bias"},
            {'params': bn, 'lr_mult': 1, 'decay_mult': 0,
             'name': "BN scale/shift"},
        ]

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        :return:
        """
        super(RGBNet, self).train(mode)
        count = 0
        if self._enable_pbn:
            print("Freezing BatchNorm2D except the first one.")
            for m in self.base_module.modules():
                if isinstance(m, nn.BatchNorm2d):
                    count += 1
                    if count >= (2 if self._enable_pbn else 1):
                        m.eval()

                        # shutdown update in frozen mode
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False



if __name__ == "__main__":
    net = RGBNet(101)