from pyexpat import features
from torchvision.datasets import ImageFolder # 1、直接加载图像数据集的目录 2、根据目标自动化分分类
from torchvision import transforms # 针对图像数据集进行一定程度的加工
from torch.utils.data import DataLoader,Dataset # DataLoader 对数据进行顺序打乱，Dataset 将数据进行批次划分
from torch.utils.data import random_split #随机按比例划分数据集
from torch import nn
import torch
from tqdm import tqdm
import numpy as np

transform = transforms.Compose([
    transforms.ToTensor()  # 将图像的数据集转化为 torch.tensor （归一化）
])
dataset = ImageFolder("./fruits", transform=transform)
train_dataset, valid_dataset = random_split(dataset, [0.8,0.2])
train_loader = DataLoader(dataset,batch_size=100,shuffle=True)
vaild_loader = DataLoader(dataset,batch_size=100)
# ================== 加载模型结构 =================
model = nn.Sequential(
    nn.Conv2d(3,16,3,1,1),
    nn.ReLU(),
    nn.Conv2d(16,32,3,1,1),
    nn.MaxPool2d(2,2),
    nn.ReLU(),
    nn.MaxPool2d(2,2),
    nn.Conv2d(32,64,3,1,1),
    nn.ReLU(),
    nn.MaxPool2d(2,2),
    nn.Flatten(),
    nn.Linear(64 * 12 * 12,1024),
    nn.ReLU(),
    nn.Linear(1024,10),
    nn.LogSoftmax(dim=-1)
)
# ============ 如果用GPU去训练模型，默认的设备是GPU
device = torch.device("cpu")
model = model.to(device)

label = dataset.classes
# ================== 加载权重 ===================
state_dict = torch.load("./save/best.pt")
model.load_state_dict(state_dict) # 模型结构加入权重
# ================ 加载数据 ========================
import cv2

image = cv2.imread("../test/1.jpg") # numpy 数据
image = cv2.resize(image,(100,100))
image = np.expand_dims(image,0) #(1,28,28,3)
image = torch.from_numpy(image).permute([0,3,1,2])
model.eval() # 启动验证模式
result = model(image.to(device).float()) #预测结果（此时结果为softmax值）
result = torch.argmax(result,dim=-1).item() # 得到最终的预测的数字内容
result = label[result]
print(result)