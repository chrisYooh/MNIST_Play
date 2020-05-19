#coding: UTF-8

import sys
import torch
import numpy as np

from importlib import import_module
from PIL import Image

def load_model():
    sys.path.append("../3_NeuralNetwork")
    module = import_module("net")
    model = module.Net()
    weight_info = torch.load("./model_weight.pth", map_location="cpu")
    model.load_state_dict(weight_info)
    model.eval()
    return model

def load_input(image_path):
    img_pil = Image.open(image_path)
    img_np = np.array(img_pil)
    img_tensor = torch.from_numpy(img_np)
    img_tensor_float = img_tensor.float()
    img_tensor_float = img_tensor_float.unsqueeze(0)
    img_tensor_float = img_tensor_float.unsqueeze(0)
    return img_tensor_float

def analysis_result(result):
    result_tensor = torch.nn.functional.softmax(result, 1);
    tarVal = torch.max(result_tensor, 1)[1]
    tarVal = tarVal.detach().numpy()[0]
    print("该手写数字是", tarVal)

# 1 加载算法模型
model = load_model();

# 2 加载输入（一张手写数字图片）
input = load_input("./test_images/1.jpeg")

# 3 执行算法（推理）
result = model.forward(input);

# 4 解析结果
analysis_result(result)
