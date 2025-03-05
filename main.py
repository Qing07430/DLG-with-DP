import os

import torch
import random

from matplotlib import pyplot as plt
from torch.utils.data import random_split, DataLoader

from dataset import get_mnist_loaders
from models import ConvNet, weights_init
from client import Client
from server import Server
from utils import plot_accuracy_curve
import matplotlib
matplotlib.use('TKAgg')

from dlg_attack import DLGAttack
from datetime import datetime

# 获取当前时间，并创建保存路径
current_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
save_dir = os.path.join('result', current_time)

# 创建文件夹
os.makedirs(save_dir, exist_ok=True)
# 参数
clients_num = 10
C = 0.2  # 参与训练的客户端比例
batch_size = 1
E = 1  # 客户端本地训练的轮数
eta = 0.01  # 学习率
rounds = 1  # 通讯轮数
sigma = 1e-5  # 噪声参数
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 模型初始化
global_model = ConvNet()
global_model.apply(weights_init)

# 数据
train_dataset, test_dataset = get_mnist_loaders()

# 按客户端数量切分
client_data = random_split(train_dataset, [len(train_dataset) // clients_num] * clients_num)

clients = [
    Client(i, global_model, DataLoader(client_data[i], batch_size=batch_size), device, eta, E, sigma)
    for i in range(clients_num)
]
# 服务器与客户端初始化
server = Server(global_model)


# 记录每轮准确率
accuracy_list = []

dlg_attacker = DLGAttack(global_model, device)

# 联邦训练
for round in range(rounds):
    print(f'Round {round + 1}/{rounds}')

    # 随机选择部分客户端
    m = max(int(C * clients_num), 1)
    selected_clients = random.sample(clients, m)

    # 下发全局模型
    global_params = server.get_global_model_params()
    for client in selected_clients:
        client.set_model_params(global_params)

    # 本地训练
    client_params_list = []
    client_data_sizes = []
    client_gradients = []  # 存梯度
    client_trained_weights = []  # 存网络参数，用于虚拟网络初始化
    client_real_images = []
    for client in selected_clients:
        local_grad, trained_weights,real_images = client.train()
        client_gradients.append(local_grad)
        client_trained_weights.append(trained_weights)
        client_params_list.append(client.get_model_params())
        client_real_images.append(real_images)
        client_data_sizes.append(len(client.train_loader.dataset))
        print(f"Client {client.id} trained")
    # 服务器聚合
    server.aggregate(client_params_list, client_data_sizes)

    # 轮到DLG攻击环节
    attacked_client_idx = random.randint(0, len(selected_clients) - 1)
    print(f"DLG Attack on Client {selected_clients[attacked_client_idx].id}")

    if round == 0:
        # 轮到DLG攻击环节
        attacked_client_idx = random.randint(0, len(selected_clients) - 1)
        print(f"DLG Attack on Client {selected_clients[attacked_client_idx].id}")

        attacked_client = selected_clients[attacked_client_idx]

        for i in range(10):  # len(client_gradients[attacked_client_idx])
            recovered_data, recovered_label = dlg_attacker.recover_data(
                client_gradients[attacked_client_idx][i],
                client_trained_weights[attacked_client_idx][i],
                client_real_images[attacked_client_idx][i],
                original_image_size=(1, 1, 28, 28),
                round_num=round,
                batch_num=i,
                save_dir=save_dir
            )
        # DLG攻击效果评价
        mse_thresholds = [0.0001, 0.00005, 0.00001, 0.000005, 0.000001]
        mse_distribution = {t: 0 for t in mse_thresholds}

        # 计算每个MSE阈值下满足条件的样本数量
        for mse in dlg_attacker.mse_list:
            for t in mse_thresholds:
                if mse <= t:
                    mse_distribution[t] += 1

        # 计算总样本数量
        total_samples = len(dlg_attacker.mse_list)

        # 将样本数量转换为比例
        mse_distribution_percentage = {t: count / total_samples for t, count in mse_distribution.items()}

        # 绘图
        plt.figure()
        plt.plot([f'≤{t}' for t in mse_thresholds], list(mse_distribution_percentage.values()), marker='o')
        plt.xlabel("MSE Threshold")
        plt.ylabel("Proportion of Batches Recovered")
        plt.title("DLG Recovery Quality Distribution")
        plt.grid(True)
        plt.show()

    # 测试
    accuracy = server.test(torch.utils.data.DataLoader(test_dataset, batch_size=batch_size), device)
    accuracy_list.append(accuracy)

    print(f'Global Model Test Accuracy: {accuracy:.4f}')

# 绘制准确率曲线
plot_accuracy_curve(accuracy_list)

print("Training Complete.")
