import numpy as np

# 加载 .npy 文件
file_path = 'predictions/confidences.npy'  # 替换为你的 .npy 文件路径
data = np.load(file_path)

# 查看形状
print(f"The shape of the npy file is: {data.shape}")
