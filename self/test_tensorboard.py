"""
@filename:test_tensorboard.py
@author:Xinpeng Qin
@time:2023--22
"""
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np

writer = SummaryWriter("logs")

img_path = "F:\\anaconda_code\\练手数据集\\train\\ants_image\\0013035.jpg"
img_PIL = Image.open(img_path)
img_array = np.array(img_PIL)
writer.add_image("test",img_array,1,dataformats='HWC')
for i in range(100):
    writer.add_scalar("y = 3x",3*i,i)
writer.close()