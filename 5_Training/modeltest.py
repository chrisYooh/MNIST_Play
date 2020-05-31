#coding: UTF-8

import os
import sys
import torch
import struct
import numpy as np

from importlib import import_module

def load_model():
    sys.path.append("../3_NeuralNetwork")
    module = import_module("net")
    model = module.Net()
    weight_info = torch.load("./Milestone/weight_epoch_10.pth", map_location="cpu")
    model.load_state_dict(weight_info)
    model.eval()
    return model

def load_data(trainingDataFilePath, trainingLabelFilePath):

    with open(trainingDataFilePath, 'rb') as imgData:
        magic, num, rows, cols = struct.unpack('>IIII', imgData.read(16))
        tdatas = np.fromfile(imgData, dtype=np.uint8).reshape(num, rows * cols)

    with open(trainingLabelFilePath, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        tlabels = np.fromfile(lbpath, dtype=np.uint8)

    tdatas_tensor = torch.from_numpy(tdatas).reshape(num, 1, rows, cols).float()
    tlabels_tensor = torch.from_numpy(tlabels).long()

    return tdatas_tensor, tlabels_tensor

def batch_test(model, batch_num, test_data, test_label, loss_func):

    model.eval()

    eval_loss = 0.
    eval_acc = 0.
    loop_num = int(len(test_data) / batch_num)

    for i in range(loop_num):

        # 1 获取 训练数据 & 训练Label
        batch_input = test_data[i * batch_num: ((i + 1) * batch_num)]
        batch_label = test_label[i * batch_num: ((i + 1) * batch_num)]

        # 2 执行推理
        batch_out = model(batch_input)

        # 3 计算模型损失
        batch_loss = loss_func(batch_out, batch_label)
        eval_loss += batch_loss.data
        # print(np.around(eval_loss.data.numpy(), decimals=3));

        # 4 模型推理准确值累加（用以计算准确率）
        pred = torch.max(batch_out, 1)[1]
        num_correct = (pred == batch_label).sum()
        eval_acc += num_correct.data

    test_num = loop_num * batch_num
    return eval_loss, eval_acc, test_num


# 1 加载模型
model = load_model();

# 2 加载测试数据
test_data, test_label = load_data("../1_MNIST/t10k-images-idx3-ubyte",
                                  "../1_MNIST/t10k-labels-idx1-ubyte")

# 3 测试准确率
loss_func = torch.nn.CrossEntropyLoss()
test_loss, test_acc, test_num = batch_test(model, 64, test_data, test_label, loss_func)
print('Test Loss: {:.6f}, Acc: {:.6f}'.format(test_loss / test_num, test_acc.float() / test_num))
