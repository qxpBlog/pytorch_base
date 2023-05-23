"""
@filename:test_transforms.py
@author:Xinpeng Qin
@time:2023/5/23  10:22
"""
from PIL import Image
from torchvision import transforms

img_path = "练手数据集/train/ants_image/0013035.jpg"
img = Image.open(img_path)
t = transforms.ToTensor()
tensor_img = t(img)
print(type(tensor_img))