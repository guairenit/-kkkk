# import numpy as np
# import torch
# import torch.nn as nn
# from model.tegcn import Model
# from graph.uav import Graph  # 正确导入 Graph 类
# import os
#
# # 设置参数
# num_class = 155  # 类别数
# num_point = 17  # 关键点数
# num_person = 2  # 每个样本中的人数
# in_channels = 3  # 输入通道数
# drop_out = 0  # Dropout 比例
# graph_args = {}  # 传递给 Graph 的参数
#
# # 加载模型
# def load_model(checkpoint_path):
#     model = Model(
#         num_class=num_class,
#         num_point=num_point,
#         num_person=num_person,
#         graph=Graph,  # 直接传递 Graph 类
#         graph_args=graph_args,
#         in_channels=in_channels,
#         drop_out=drop_out
#     )
#     model.load_state_dict(torch.load(checkpoint_path))
#     model.eval()  # 切换到评估模式
#     return model
#
# # 读取数据
# def load_data(data_path):
#     return np.load(data_path)
#
# # 主函数
# if __name__ == '__main__':
#     checkpoint_path = 'runs/2101-56-10398.pt'
#     data_path = 'data/uav/xsub1/test_data.npy'
#
#     # 加载模型和数据
#     model = load_model(checkpoint_path)
#     test_data = load_data(data_path)
#
#     # 转换数据为 PyTorch 张量并调整形状
#     test_tensor = torch.tensor(test_data, dtype=torch.float32)
#
#     # 进行预测
#     with torch.no_grad():
#         outputs = model(test_tensor)
#
#     # 获取置信度
#     probabilities = nn.Softmax(dim=1)(outputs)  # 计算置信度
#
#     # 保存置信度文件
#     output_path = 'predictions/confidences.npy'
#     np.save(output_path, probabilities.numpy())
#
#     print(f'预测结果已保存至 {output_path}')


import numpy as np
import torch
import torch.nn as nn
from model.tegcn import Model
from graph.uav import Graph  # 正确导入 Graph 类
import os

# 设置参数
num_class = 155  # 类别数
num_point = 17  # 关键点数
num_person = 2  # 每个样本中的人数
in_channels = 3  # 输入通道数
drop_out = 0  # Dropout 比例
graph_args = {}  # 传递给 Graph 的参数
batch_size = 4  # 设置批量大小

# 检查是否有可用的GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 加载模型
def load_model(checkpoint_path):
    model = Model(
        num_class=num_class,
        num_point=num_point,
        num_person=num_person,
        graph=Graph,  # 直接传递 Graph 类
        graph_args=graph_args,
        in_channels=in_channels,
        drop_out=drop_out
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))  # 将模型加载到相应的设备
    model.to(device)  # 将模型转移到GPU
    model.eval()  # 切换到评估模式
    return model

# 读取数据
def load_data(data_path):
    return np.load(data_path)

# 主函数
if __name__ == '__main__':
    checkpoint_path = 'runs/2101-56-10398.pt'
    data_path = 'data/uav/xsub1/test_data.npy'

    # 加载模型和数据
    model = load_model(checkpoint_path)
    test_data = load_data(data_path)

    # 转换数据为 PyTorch 张量并调整形状
    test_tensor = torch.tensor(test_data, dtype=torch.float32).to(device)  # 将数据转移到GPU

    # 切分数据为小批次
    num_samples = test_tensor.shape[0]
    num_batches = (num_samples + batch_size - 1) // batch_size  # 计算总批次数

    all_probabilities = []

    # 进行预测
    with torch.no_grad():
        for i in range(num_batches):
            batch_data = test_tensor[i * batch_size:(i + 1) * batch_size]  # 提取当前批次数据
            outputs = model(batch_data)  # 前向传播
            probabilities = nn.Softmax(dim=1)(outputs)  # 计算置信度
            all_probabilities.append(probabilities.cpu().numpy())  # 转移到CPU以便保存

    # 拼接所有批次的置信度
    all_probabilities = np.concatenate(all_probabilities, axis=0)

    # 保存置信度文件
    output_path = 'predictions/confidences.npy'
    np.save(output_path, all_probabilities)

    print(f'预测结果已保存至 {output_path}')
