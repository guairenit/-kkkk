# 现在我的tegcn模型已经训练好了，权重参数位于runs/56.pt,模型位于model/tegcn.py ,数据位于data/test_data.npy,现在我的预测文件放在predictions/predict.py里面，我要编写代码进行预测数据集b，得到置信度文件，我该如何编写代码


import torch
import numpy as np
import sys
import os
import sys
import os

# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 添加模型目录到 sys.path
model_path = os.path.join(current_dir, '../model')
sys.path.append(model_path)

# 导入模型类
from tegcn import Model
from module_ta import Multi_Head_Temporal_Attention





def load_model(model_path, device):
    model = Model(num_class=155, num_point=17, num_person=2, graph='graph.uav.Graph')
    model = model.to(device)

    # 加载训练好的权重
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)

    return model


def load_data(data_path):
    data = np.load(data_path)
    data = torch.tensor(data, dtype=torch.float32)
    return data


def predict(model, data, device):
    model.eval()
    data = data.to(device)
    with torch.no_grad():
        output = model(data)
        confidence = torch.softmax(output, dim=1)
    return confidence


def save_results(confidence, save_path):
    np.save(save_path, confidence.cpu().numpy())


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 设置路径
    model_path = './runs/56.pt'
    data_path = './data/test_data.npy'
    save_path = './predictions/confidence_results.npy'

    # 加载模型和数据
    model = load_model(model_path, device)
    data = load_data(data_path)

    # 进行预测
    confidence = predict(model, data, device)

    # 保存预测结果
    save_results(confidence, save_path)
    print(f"Prediction saved to {save_path}")


if __name__ == "__main__":
    main()
