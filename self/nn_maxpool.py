"""
@filename:nn_maxpool.py
@author:Xinpeng Qin
@time:2023/5/24  11:27
"""
import torch
import torchvision.datasets
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("./data",train=False,transform=torchvision.transforms.ToTensor()
                                       ,download=False)
dataloader = DataLoader(dataset,batch_size=64)

class Mynn(nn.Module):
    def __init__(self):
        super(Mynn,self).__init__()
        self.maxpool1 = MaxPool2d(3,ceil_mode=True)

    def forward(self, input):
        output = self.maxpool1(input)
        return output

mynn = Mynn()
step = 0
writer = SummaryWriter("logs")
for data in dataloader:
    imgs,target = data
    writer.add_images("maxpool_test_input",imgs,step)
    output = mynn(imgs)
    writer.add_images("maxpool_test_output",output,step)
    step += 1