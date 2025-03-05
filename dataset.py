import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Subset


def get_mnist_loaders():
    transform = transforms.ToTensor()

    train_dataset = datasets.MNIST(root='data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='data', train=False, download=True, transform=transform)

    # 只使用10%的数据
    train_size = int(0.2 * len(train_dataset))
    test_size = int(0.2 * len(test_dataset))

    train_indices = torch.randperm(len(train_dataset))[:train_size]
    test_indices = torch.randperm(len(test_dataset))[:test_size]

    train_subset = Subset(train_dataset, train_indices)
    test_subset = Subset(test_dataset, test_indices)

    return train_subset, test_subset

