# -*- coding: utf-8 -*-
# @Time    : 2019/5/24 16:55
# @Author  : zwenc
# @File    : dataset.py

from torch.utils import data
import cv2
import numpy as np

from torch.utils.data import TensorDataset
import torch

# data.Dataset 继承父类，就可以加入到torch家族，使用它内部的其他功能
class ListDataSet(data.Dataset):
    def __init__(self, DataPath, Train = True, class_num = 101):
        """
        :param DataPath: 数据集根目录地址
        :param Train:    是否加载训练集，默认加载
        :return:         Null
        """
        self.Train = Train
        self.class_num = class_num
        if self.Train:
            # trainlist01文件里面自带ID信息
            self._GetAllTrainData(DataPath + "/ucfTrainTestlist/trainlist01.txt")
        else:
            # 因为testlist01文件没有ID信息，所以需要先获取ClassID信息
            self._GetClassID(DataPath + "/ucfTrainTestlist/classInd.txt")
            self._GetAllTestData(DataPath + "/ucfTrainTestlist/testlist01.txt")

    def __getitem__(self, item): # 来自继承的接口，必须实现，供上层调用
        """
        :param item: 上层传入想要的index，类型为int
        :return: Data,Lable
        """
        if self.Train:
            return self.TrainData[item][0],self.TrainData[item][1]
        else:
            return self.TestData[item][0],self.TestData[item][1]

    def __len__(self):  # 来自继承的接口，必须实现，供上层调用
        if self.Train:
            # print("a:",len(self.TrainData))
            return len(self.TrainData)
        else:
            # print("a:",len(self.TestData))
            return len(self.TestData)

    def _GetAllTrainData(self, TrainDataPath):
        self.TrainData = list()
        TrainDataFile1 = open(TrainDataPath,"r")

        for line in TrainDataFile1:
            temp = line.strip("\n").split(" ")
            temp[1] = int(temp[1]) - 1
            # temp: ['ApplyEyeMakeup/v_ApplyEyeMakeup_g08_c01.avi', 1]
            if temp[1] >= self.class_num: # 选择训练集lable的数量，便于在初期进行调试
                break
            self.TrainData.append(temp)

        TrainDataFile1.close()
        #print(self.TrainData)

    def _GetAllTestData(self, TestDataPath):
        self.TestData = list()
        TestDataFile1 = open(TestDataPath,"r")

        for line in TestDataFile1:
            linedata = line.strip("\n") # 去掉最后的换行符
            # linedata: ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.avi
            classid = self.ClassID[linedata.split("/")[0]] # 获得该视频的分类

            if classid >= self.class_num: # 选择训练集lable的数量，便于在初期进行调试
                break
            temp = [linedata,classid]   # 制作成一个组
            # temp: [1, 'ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.avi']
            self.TestData.append(temp) # 添加到list
        TestDataFile1.close()

    # 由于Trainlist文件自带了ID，所以只有test数据才用的上查字典
    def _GetClassID(self, ClassIDPath):
        self.ClassID = dict()
        classidfile = open(ClassIDPath,"r")
        if classidfile == None:
            print("ClassPath"+"文件打开失败")
            raise IOError
        for line in classidfile:
            # 读取文件的每一行，去掉字符，以空格分开数据
            temp = line.strip("\n").split(" ")
            # temp: ['1', 'ApplyEyeMakeup']
            temp[0] = int(temp[0]) # 读出来的是str类型，强转为int类型
            # temp: [1, 'ApplyEyeMakeup']
            # 做成字典
            self.ClassID[temp[1]] = temp[0] - 1
        classidfile.close()
        #print(self.ClassID)  # 可打印查看列表数据

class myDataSet(data.Dataset):
    # 这里要把列表信息处理好
    def __init__(self, DataPath, class_num = 101,Train=True, ModuleType="RGB", Segments=3,
                 DataLen = 1,Weight = 224, Height = 224):
        self.DataPath = DataPath      # 数据集的位置
        self.Train = Train            # 是否选择训练集
        self.moduleType = ModuleType  # 输出模式，RGB，RGBDff,光流
        self.segments = Segments      # 视频分为几段
        self.weight = Weight          # 需要输出视频的宽
        self.height = Height          # 需要输出视频的高
        self.DataLen = DataLen        # 每一个Segment提取多少组数据
        self.class_num = class_num    # 拿多少类用来学习，一般用于调试，其他时候采用默认值

        # 不同类型的通道数不一样。光流通道数为2（x,y方向的矢量各一个图片）
        if self.moduleType == "RGB":
            self.channel = 3

        # 获取视频列表数据信息
        self.data = ListDataSet(self.DataPath,self.Train,self.class_num)

    # item其实就是index数组下标，这里是数据的输出出口，__getitem__是内联函数
    def __getitem__(self, item):
        VideoDataPath = self.data[item][0]
        Label = self.data[item][1]

        # 获取数据的时候，开始对视频进行处理，这样虽然浪费的时间，但是节约了内存
        if self.moduleType == "RGB": # 为了后期拓展，加个类型判断
            images = self._Change_to_RGB(self.DataPath + "/UCF101/" + VideoDataPath)

        # 转换为torch参数类型输出，torch网络只能使用torch内部的类型
        return torch.from_numpy(images),torch.full([self.segments*self.DataLen],Label)

    def _Change_to_RGB(self, VideoDataPath):
        """
        :param VideoDataPath:  视频地址   # F:/data/UCF101/PushUps/v_PushUps_g01_c01.avi
        :return: RGB数据
        """
        cap = cv2.VideoCapture()  # 初始化opencv
        cap.open(VideoDataPath)   # 打开视频文件
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 计算帧的数量
        segment_length = int(n_frames / self.segments)     # 计算没一段帧的数量
        # print("n_frames ",n_frames)
        # print("segment_length ",segment_length)
        # print("DataLen = ",self.DataLen)
        if self.DataLen > segment_length:
            print("所需要的数据长度大于每段所拥有的数据长度")
            raise IOError

        # dstimage 初始化为0，主要是开辟空间，初始化为1也没有问题
        dstimage = np.zeros([self.DataLen*self.segments,self.channel,self.weight,self.height])
        times = 0

        for i in range(self.segments):
            # 对每个段都重新计算抽取帧的位置
            temp = self._generate_random(segment_length,self.DataLen)  # 在每一segment里面随机获得self.DataLen张图片，生成index

            for k in temp:
                # 抽取指定位置的图片
                res,image = cap.read(k + i * segment_length)  # image.shape : (240, 320, 3)
                if not res: # 判断是否抽取成功
                    break

                # 由于网络输入的图片尺寸是固定的，不方便修改，这里修改训练库图片尺寸（总不能为每个分辨率设计一个网络吧）
                image = cv2.resize(image,(self.weight,self.height),interpolation = cv2.INTER_AREA)  # 修改图片尺寸
                # shape: [224,224,3]
                # 网络无法使用上面的shape，所以修改至下面的
                image = self._Image_ReShape(image)  # shape: [3,224,224]

                # 添加到输出空间中
                dstimage[times] = image
                times += 1

        # 释放opencv空间
        cap.release()
        # print(dstimage.shape)
        return dstimage

    def _generate_random(self, num_range, nsize):
        """
        Args: 生成nsize个，0~num_range范围的不同随机数的list，且随机数不能重复
            num_range: 需要的随机数范围
            nsize:     需要的随机数数量

        Returns: 随机数组成的list

        """
        generate = True
        temp = list()  # type: list
        while generate:
            temp = np.random.randint(num_range,size=nsize).tolist()  # type: list
            temp.sort()

            if nsize == 1:
                break
            # 判断是否有重复的帧序号
            for i in range(nsize - 1):
                if temp[i] == temp[i + 1]:
                    generate = True
                    break
                generate = False
        # print(temp)
        return temp

    def _Image_ReShape(self, image):
        """
        Args:
            image: 输入图片 [224,224,3]

        Returns: reshape后的图片 [3,224,224]
        """
        dstimage = np.zeros([self.channel,self.weight,self.height])
        for a in range(self.channel):
            dstimage[a] = image[:,:,a]

        return dstimage

    # 返回长度
    def __len__(self):
        return self.data.__len__()


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    a = myDataSet("F:/data",Train=False,DataLen=5)

    # 分组，并且打乱数据
    b = data.DataLoader(dataset=a,batch_size=1,shuffle=True)
    for index,(D,L) in enumerate(b):
        print(D.size())
        print(" ")
        plt.imshow(D[0][0][0])
        plt.show()
        if index == 5:
            break

        # print(D)
        # print(L)


