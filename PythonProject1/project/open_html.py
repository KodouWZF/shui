from flask import Flask
from flask import render_template
from flask import request
import torch
import cv2
import numpy as np
from flask_cors import CORS

from torchvision.datasets import ImageFolder # 1、直接加载图像数据集的目录 2、根据目标自动化分分类
from torch.utils.data import DataLoader,Dataset # DataLoader 对数据进行顺序打乱，Dataset 将数据进行批次划分
from torchvision import transforms # 针对图像数据集进行一定程度的加工

app = Flask(__name__)
CORS(app)

# ========================加载模型结构=============================
from torch import nn

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
# ===========================加载权重 ==========================
state_dict = torch.load("./save/best.pt")
model.load_state_dict(state_dict)  # 模型结构加入权重

transform = transforms.Compose([
    transforms.ToTensor()  # 将图像的数据集转化为 torch.tensor
])
dataset = ImageFolder("./fruits", transform=transform)
label = dataset.classes
# ---- 渲染整个识别的网页-----
@app.route("/")
def detect_number():
    return render_template("index.html")


# ---- 上传文件，并识别该内容 ----
@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['the_file']  # 获取文件对象
        f.save('./static/upload/1.png')  # 将文件对象保存到服务器
        # ---- 识别本地文件--------
        img = cv2.imread("static/upload/1.png")
        # img = cv2.resize(img,(28,28))
        img = np.expand_dims(img, 0)
        img = torch.from_numpy(img)
        img = torch.permute(img, [0, 3, 1, 2])
        model_cpu = model.to(torch.device("cpu"))
        predict = model_cpu(img.float())
        result = torch.argmax(predict, dim=-1)
        result = label[result]

    return {"code": 202, "result": result}

#开始
if __name__ == "__main__":
    app.run("192.168.235.201", 9000, debug=True)
