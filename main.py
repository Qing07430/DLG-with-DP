import torch
import random

from torch.utils.data import random_split, DataLoader

from dataset import get_mnist_loaders
from models import ConvNet, weights_init
from client import Client
from server import Server
from utils import plot_accuracy_curve
import matplotlib
matplotlib.use('Agg')

from dlg_attack import DLGAttack

# 参数
clients_num = 10
C = 0.2  # 参与训练的客户端比例
batch_size = 1
E = 1  # 客户端本地训练的轮数
eta = 0.01  # 学习率
rounds = 1  # 通讯轮数
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 模型初始化
global_model = ConvNet()
global_model.apply(weights_init)

# 数据
train_dataset, test_dataset = get_mnist_loaders()

# 按客户端数量切分
client_data = random_split(train_dataset, [len(train_dataset) // clients_num] * clients_num)

clients = [
    Client(i, global_model, DataLoader(client_data[i], batch_size=batch_size), device, eta, E)
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
    for client in selected_clients:
        local_grad, trained_weights = client.train()
        client_gradients.append(local_grad)
        client_trained_weights.append(trained_weights)
        client_params_list.append(client.get_model_params())
        client_data_sizes.append(len(client.train_loader.dataset))
        print(f"Client {client.id} trained")
    # 服务器聚合
    server.aggregate(client_params_list, client_data_sizes)

    # 轮到DLG攻击环节
    attacked_client_idx = random.randint(0, len(selected_clients) - 1)
    print(f"DLG Attack on Client {selected_clients[attacked_client_idx].id}")

    attacked_client = selected_clients[attacked_client_idx]

    for i in range(3): # len(client_gradients[attacked_client_idx])
        recovered_data, recovered_label = dlg_attacker.recover_data(
            client_gradients[attacked_client_idx][i],
            client_trained_weights[attacked_client_idx][i],
            original_image_size=(1, 1, 28, 28)  # 具体看你的数据形状
        )

    # 测试
    accuracy = server.test(torch.utils.data.DataLoader(test_dataset, batch_size=batch_size), device)
    accuracy_list.append(accuracy)

    print(f'Global Model Test Accuracy: {accuracy:.4f}')

# 绘制准确率曲线
plot_accuracy_curve(accuracy_list)

print("Training Complete.")

