import torch
import random
from torch.utils.data import DataLoader, Subset
from dataset import get_mnist_loaders
from models import ConvNet, weights_init
from client import Client
from server import Server
from utils import plot_accuracy_curve, dirichlet_partition, WrapperDataset, simple_partition
import matplotlib
matplotlib.use('Agg')
from dlg_attack import DLGAttack
import os
import time
from datetime import datetime

# 获取当前时间，并创建保存路径
current_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
save_dir = os.path.join('result', current_time)

# 创建文件夹
os.makedirs(save_dir, exist_ok=True)

# 参数
clients_num = 50
C = 0.5  # 参与训练的客户端比例
batch_size = 1
E = 5  # 客户端本地训练的轮数
eta = 0.01  # 学习率
rounds = 10  # 通讯轮数
sigma = 0.3  # 噪声参数
alpha = 10.0 # Dirichlet参数,控制non-IID程度
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 模型初始化
global_model = ConvNet()
global_model.apply(weights_init)

# 数据
train_dataset, test_dataset = get_mnist_loaders()
'''
client_data_indices = dirichlet_partition(train_dataset, clients_num, alpha)
# 按客户端数量切分
client_loaders = []
for i in range(clients_num):
    client_subset = Subset(train_dataset, client_data_indices[i])
    wrapped_dataset = WrapperDataset(client_subset)  # 确保格式 (image, label)
    loader = DataLoader(wrapped_dataset, batch_size=batch_size, shuffle=True)
    client_loaders.append(loader)

clients = [
    Client(i, global_model, client_loaders[i], device, eta, E, sigma)
    for i in range(clients_num)
]
'''
client_data_indices = simple_partition(train_dataset, clients_num)

client_loaders = []
for i in range(clients_num):
    client_subset = Subset(train_dataset, client_data_indices[i])
    wrapped_dataset = WrapperDataset(client_subset)  # 确保格式 (image, label)
    loader = DataLoader(wrapped_dataset, batch_size=batch_size, shuffle=True)
    client_loaders.append(loader)

clients = [
    Client(i, global_model, client_loaders[i], device, eta, E, sigma)
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
        #modified_params = {key.replace('_module.', ''): value for key, value in global_params.items()}
        client.set_model_params(global_params)

        # 本地训练
    client_params_list = []
    client_data_sizes = []
    client_gradients = []  # 存梯度
    client_trained_weights = []  # 存网络参数，用于虚拟网络初始化
    for client in selected_clients:
        local_grad, trained_weights = client.train()
        client_gradients.append(local_grad)
        client_trained_weights.append(trained_weights)
        client_params = client.get_model_params()
        # 去除 '_module' 前缀
        client_params = {key.replace('_module.', ''): value for key, value in client_params.items()}
        client_params_list.append(client_params)
        client_data_sizes.append(len(client.train_loader.dataset))
        print(f"Client {client.id} trained")
    # 服务器聚合
    server.aggregate(client_params_list, client_data_sizes)

    epsilons = [client.get_privacy_spent() for client in selected_clients]
    avg_epsilon = sum(epsilons) / len(epsilons)
    print(f"Round {round + 1}: Average ε = {avg_epsilon:.4f}")

    # 轮到DLG攻击环节
    attacked_client_idx = random.randint(0, len(selected_clients) - 1)
    print(f"DLG Attack on Client {selected_clients[attacked_client_idx].id}")

    attacked_client = selected_clients[attacked_client_idx]

    for i in range(3): # len(client_gradients[attacked_client_idx])
        recovered_data, recovered_label = dlg_attacker.recover_data(
            client_gradients[attacked_client_idx][i],
            client_trained_weights[attacked_client_idx][i],
            original_image_size=(1, 1, 28, 28),  # 具体看你的数据形状
            round_num=round,
            batch_num=i,
            save_dir=save_dir
        )

    # 测试
    accuracy = server.test(torch.utils.data.DataLoader(test_dataset, batch_size=batch_size), device)
    accuracy_list.append(accuracy)

    print(f'Global Model Test Accuracy: {accuracy:.4f}')

# 绘制准确率曲线
plot_accuracy_curve(accuracy_list)

print("Training Complete.")

