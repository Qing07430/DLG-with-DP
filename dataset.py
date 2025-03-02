import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import random_split, DataLoader, Subset


def get_mnist_loaders(batch_size=1):
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(root='data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='data', train=False, download=True, transform=transform)

    # 裁剪数据，随机选取20%的数据
    train_size = int(0.1 * len(train_dataset))  # 20%
    train_indices = torch.randperm(len(train_dataset))[:train_size]
    train_subset = Subset(train_dataset, train_indices)

    test_size = int(0.1 * len(test_dataset))
    test_indices = torch.randperm(len(test_dataset))[:test_size]
    test_subset = Subset(test_dataset, test_indices)

    return train_subset, test_subset
