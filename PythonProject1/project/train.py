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
    transforms.ToTensor()  # 将图像的数据集转化为 torch.tensor
])
dataset = ImageFolder("./fruits", transform=transform)
train_dataset, valid_dataset = random_split(dataset, [0.8,0.2])
train_loader = DataLoader(dataset,batch_size=100,shuffle=True)
vaild_loader = DataLoader(dataset,batch_size=100)

# 建立100 * 100的训练模型
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

# ==================== 训练 ======================
criterion = nn.CrossEntropyLoss() # 交叉熵损失 (softmax + 交叉熵)
# --------------- 优化器 --------------------
optimizer = torch.optim.SGD(model.parameters(),lr=0.01, momentum=0.9)
# ------------- 开始训练 -------------------
best_acc = 0

epochs = 200
for epoch in range(epochs):
    print(f"{epoch + 1} / {epochs}")
    # 开启训练模式
    model.train()
    loss_list = []
    acc_list = []
    loop1 = tqdm(train_loader)
    # 将数据集按照批次进行运算
    for data in loop1:
        features, labels = data
        optimizer.zero_grad()  # 对所有的求导清零
        labels_predict = model(features.float())  # 用模型预测结果
        loss = criterion(labels_predict, labels.long())  # MSE均方差要求两个参数的矩阵大小一致
        loss.backward()  # 开启参数求导
        optimizer.step()  # 更新所有求导的w，b的值，进行下一轮训练
        loss_list.append(loss.item())
        loop1.set_description(f"train_loss:{loss.item():.4f}")
    # 开启验证模式
    model.eval()
    loop2 = tqdm(vaild_loader)
    for data in loop2:
        features, labels = data
        labels_predict = torch.argmax(features.float())
        accuracy = sum(labels_predict == labels) / len(labels)
        acc_list.append(accuracy)
        loop2.set_description(f"valid_acc:{accuracy * 100:.2f}")

    total_loss = sum(loss_list) / len(loss_list)
    total_acc = sum(acc_list) / len(acc_list)
    print(f"total_loss:{total_loss:.4f} -- total_acc:{total_acc * 100:.2f}")

    if best_acc < total_acc:
        # 保存模型(此处仅保存模型的权重（w, b）,会加快保存进程)
        torch.save(model.state_dict(), "./save/best.pt")
        best_acc = total_acc