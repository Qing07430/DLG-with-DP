import torch
import torch.nn.functional as F

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


