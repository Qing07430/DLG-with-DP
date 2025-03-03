import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def label_to_onehot(target, num_classes=10):
    target = torch.unsqueeze(target, 1)
    onehot_target = torch.zeros(target.size(0), num_classes, device=target.device)
    onehot_target.scatter_(1, target, 1)
    return onehot_target

def cross_entropy_for_onehot(pred, target):
    return torch.mean(torch.sum(- target * F.log_softmax(pred, dim=-1), 1))

# 画图
import matplotlib.pyplot as plt
import os

def plot_accuracy_curve(accuracy_list, save_path='accuracy_curve.png'):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(accuracy_list) + 1), accuracy_list, marker='o', linestyle='-')
    plt.xlabel('Communication Round')
    plt.ylabel('Test Accuracy')
    plt.title('Global Model Test Accuracy Across Rounds')
    plt.grid(True)
    plt.savefig(save_path)

    os.startfile(save_path)

# 可视化函数
def visualize_reconstruction(data):
    plt.figure()
    plt.imshow(data[0].cpu().detach().numpy().squeeze(), cmap='gray')
    plt.title("Reconstructed Image")
    plt.axis('off')
    plt.show()

# 普通数据划分
def simple_partition(train_dataset, clients_num):
    data_size = len(train_dataset)
    indices = torch.randperm(data_size)  # 随机打乱索引
    split_size = data_size // clients_num

    client_indices = [
        indices[i * split_size : (i + 1) * split_size].tolist()
        for i in range(clients_num)
    ]

    # 如果不能整除，把余下的数据分给最后一个客户端
    if data_size % clients_num != 0:
        client_indices[-1].extend(indices[clients_num * split_size:].tolist())

    return client_indices


# Dirichlet非IID数据划分函数
def dirichlet_partition(dataset, num_clients, alpha=0.5):
    """
    Dirichlet non-IID partitioning of dataset.
    Returns a list of indices for each client.
    """
    num_samples = len(dataset)
    labels = torch.tensor([dataset[i][1] for i in range(num_samples)])  # 全部标签

    # 初始化每个client的数据索引列表
    client_indices = [[] for _ in range(num_clients)]

    # 针对每个类别，按照Dirichlet分布随机分配给不同client
    num_classes = 10  # MNIST共10类
    for c in range(num_classes):
        class_indices = torch.where(labels == c)[0].numpy()
        np.random.shuffle(class_indices)

        # 按Dirichlet分布给每个client分配样本
        proportions = np.random.dirichlet(alpha=alpha * np.ones(num_clients))
        proportions = (proportions * len(class_indices)).astype(int)

        # 确保分配总数与类别样本数一致
        proportions[-1] = len(class_indices) - sum(proportions[:-1])

        # 分配数据到各个client
        start = 0
        for client_id in range(num_clients):
            client_indices[client_id].extend(class_indices[start:start + proportions[client_id]])
            start += proportions[client_id]

    return client_indices

class WrapperDataset(Dataset):
    """ 确保数据格式始终为 (image, label) """
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        return image, label

    def __len__(self):
        return len(self.dataset)