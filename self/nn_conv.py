"""
@filename:nn_conv.py
@author:Xinpeng Qin
@time:2023/5/24  10:26
"""
import torch
import torchvision
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("./data",train=False,transform=torchvision.transforms.ToTensor()
                                       ,download=True)
dataloader = DataLoader(dataset,batch_size=64)

class Mynn(nn.Module):
    def __init__(self):
        super(Mynn,self).__init__()
        self.conv1 = Conv2d(3,6,kernel_size=3,stride=1,padding=0)

    def forward(self, input):
        output = self.conv1(input)
        return output

mynn = Mynn()

writer = SummaryWriter("./logs")
step = 0
for data in dataloader:
    imgs,target = data
    output = mynn(imgs)
    writer.add_images("input",imgs,step)
    output = torch.reshape(output,(-1,3,30,30))
    writer.add_images("output",output,step)
    step += 1