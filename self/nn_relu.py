"""
@filename:nn_relu.py
@author:Xinpeng Qin
@time:2023/5/24  12:01
"""
import torch
from torch import nn
from torch.nn import ReLU

input = torch.tensor([[1,-0.5],
                      [-1,3]])

intput = torch.reshape(input,(-1,1,2,2))


class Mynn(nn.Module):
    def __init__(self):
        super(Mynn, self).__init__();
        self.relu1 = ReLU()

    def forward(self,input):
        output = self.relu1(input)
        return output

mynn = Mynn()

output = mynn(input)
print(output)