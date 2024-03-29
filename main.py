                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   # -*- coding: utf-8 -*-
# @Time    : 2019/5/24 16:55
# @Author  : zwenc
# @File    : main.py

import torch
from RGBNet import  RGBNet
from dataset import myDataSet
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.autograd import Variable
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = "0"

device_ids = [0,1]

learn_rate = 0.01
learn_epoch = 100
class_num = 2          # 设置两个学习lable，最大为101
learn_momentum = 0

net = RGBNet(num_class=class_num).cuda()

train_data = myDataSet("F:/data",DataLen=1,class_num=class_num)
train_data_len = train_data.__len__()
print("train_data_len: %d"%train_data_len)

test_data = myDataSet("F:/data",DataLen=1,class_num=class_num,Train=False)
test_data_len = test_data.__len__()
print("test_data_len: %d"%test_data_len)

train_mini_batch = DataLoader(dataset=train_data,batch_size=1,shuffle=True)
test_mini_batch = DataLoader(dataset=test_data,batch_size=1,shuffle=True)

criterion = torch.nn.CrossEntropyLoss().cuda()

optimizer = SGD(net.get_optim_policies(),lr=learn_rate,momentum=learn_momentum)

# print(list(net.parameters())
# for a in net.named_modules():
#     print(a)
    # print(a.size())
# print(net)

if os.path.exists("net_parameters.pkl"):
    net.load_state_dict(torch.load("net_parameters.pkl"))

import time
for epoch in range(learn_epoch):
    print(time.asctime(time.localtime(time.time())))
    envLoss = float(0)
    # time1 = 0
    # time2 = time.clock()
    net.train()
    for index,(D,L) in enumerate(train_mini_batch):
        # time1 = time.clock()
        # temp1 = time1 - time2
        # print("begin")
        OD = Variable(D[0].float(),requires_grad=True).cuda()
        OL = Variable(L[0].long()).cuda()
        output = net(OD)           # type: torch.Tensorprint(D.size())

        # output : torch.Size([3, 101])
        # OL     : torch.Size([3])
        a,b = torch.max(output.mean(dim=0),dim=0)
        loss = criterion(output,OL)  # type: torch.Tensor
        # print("index: %d,loss：%f, ofileindex: %d ,ofileindex: %d"%(index,loss.data.item(),OL[0],b))
        envLoss += loss.data.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # time2 = time.clock()
        # print("获取数据用时：",temp1)
        # print("训练用时：",time2-time1)
        # print("训练时间/获取数据时间：",(time2 - time1)/temp1)
        # print("end")

    print("epoch:%d, 训练平均损失为：%f"%(epoch, envLoss/train_data_len))

    torch.save(net.state_dict(),"net_parameters.pkl")

    net.eval()
    ture_count = 0
    ture_loss = float(0)
    for index,(D,L) in enumerate(test_mini_batch):
        OD = Variable(D[0].float(),requires_grad=True).cuda()
        OL = Variable(L[0].long()).cuda()
        output = net(OD)           # type: torch.Tensorprint(D.size())

        a,b = torch.max(output.mean(dim=0),dim=0)
        loss = criterion(output,OL)  # type: torch.Tensor
        ture_loss += loss.data.item()

        if b == OL[0]:
            ture_count += 1

    print("正确率：%f,测试平均损失：%f"%(ture_count/test_data_len,ture_loss/test_data_len))


    # optimizer.zero_grad()
