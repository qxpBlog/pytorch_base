"""
@filename:nn_func_conv.py
@author:Xinpeng Qin
@time:2023/5/24  9:40
"""
import torch

input = torch.tensor([[1,2,0,3,1],
                      [0,1,2,3,1],
                      [1,2,1,0,0],
                      [5,2,3,1,1],
                      [2,1,0,1,1]])
kernel = torch.tensor([[1,2,1],
                       [0,1,0],
                       [2,1,0]])
input = torch.reshape(input,(1,1,5,5))
kernel = torch.reshape(kernel,(1,1,3,3))
output = torch.nn.functional.conv2d(input,kernel,stride=1,padding=1)
print(output)