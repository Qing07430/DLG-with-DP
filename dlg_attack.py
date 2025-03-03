import os
import copy
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch import optim
import numpy as np
from utils import cross_entropy_for_onehot as criterion
from torchvision import transforms
import matplotlib
matplotlib.use('TkAgg')
To_image = transforms.ToPILImage()



class DLGAttack:
    def __init__(self, target_model, device):
        self.target_model = target_model
        self.device = device

    def recover_data(self, target_gradient, client_trained_weights, original_image_size,round_num, batch_num, save_dir):

        grad_norms = []
        for layer_grad in target_gradient:
            # 计算每个参数的L2范数
            layer_param_grad_norms = [param_grad.view(-1).norm().item() for param_grad in layer_grad]
            grad_norms.append(layer_param_grad_norms)

        # 展平梯度范数列表
        flat_grad_norms = [norm for layer_norms in grad_norms for norm in layer_norms]

        print(len(flat_grad_norms))

        # 选择L2范数最高的前12.5%的参数梯度
        #num_select = len(flat_grad_norms) // 8  # 一共90个左右的参数梯度
        sorted_indices_global = np.argsort(flat_grad_norms)[-92:]

        # 初始化虚拟数据，确保requires_grad=True
        history = []
        dummy_data = torch.randn(original_image_size, requires_grad=True, device=self.device)
        dummy_label = torch.randn(10, requires_grad=True, device=self.device)

        # 使用LBFGS优化器
        optimizer = optim.LBFGS([dummy_data, dummy_label], lr=1, history_size=100, max_iter=20)

        dummy_model = copy.deepcopy(self.target_model).to(self.device)
        dummy_model.load_state_dict(client_trained_weights)


        for iters in range(300):
            def closure():
                optimizer.zero_grad()

                # 计算虚拟数据的梯度
                dummy_pred = dummy_model(dummy_data)
                dummy_onehot_label = F.softmax(dummy_label, dim=-1)
                dummy_loss = criterion(dummy_pred,dummy_onehot_label)
                dummy_gradients = torch.autograd.grad(dummy_loss, dummy_model.parameters(), create_graph=True)

                # 获取选定参数梯度的索引
                selected_dummy_dy_dx = []
                selected_original_dy_dx = []

                # 追踪全局索引
                global_index = 0

                # 遍历每层的梯度
                for layer_dummy_grad, layer_original_grad in zip(dummy_gradients, target_gradient):
                    layer_selected_dummy_grad = []
                    layer_selected_original_grad = []

                    # 对每层的参数梯度
                    for param_dummy_grad, param_original_grad in zip(layer_dummy_grad, layer_original_grad):
                        # 检查当前参数的全局索引是否在选定索引中
                        if global_index in sorted_indices_global:
                            layer_selected_dummy_grad.append(param_dummy_grad.view(-1))
                            layer_selected_original_grad.append(param_original_grad.view(-1))

                        global_index += 1

                    # 如果该层有选定的梯度，则堆叠
                    if layer_selected_dummy_grad:
                        selected_dummy_dy_dx.append(torch.cat(layer_selected_dummy_grad))
                        selected_original_dy_dx.append(torch.cat(layer_selected_original_grad))

                # 计算选定梯度之间的差距
                grad_diff = 0
                for gx, gy in zip(selected_dummy_dy_dx, selected_original_dy_dx):
                    grad_diff += ((gx - gy) ** 2).sum()  # 计算梯度之间的差异

                grad_diff.backward()  # 执行反向传播

                return grad_diff

            optimizer.step(closure)
            if iters % 10 == 0:
                history.append(To_image(dummy_data[0].cpu()))

        self.save_recovered_images(history, round_num, batch_num, save_dir)

        return dummy_data, dummy_label

    def save_recovered_images(self, history, round_num, batch_num, save_dir):
        plt.figure(figsize=(12, 5))
        for i in range(30):
            plt.subplot(3, 10, i + 1)
            plt.imshow(history[i])
            plt.title(f"iter={i * 10}")
            plt.axis('off')

        # 保存图片
        save_path = os.path.join(save_dir, f'round_{round_num + 1}_batch_{batch_num + 1}.png')
        plt.savefig(save_path)
        plt.close()
        print(f"Saved DLG recovery image to {save_path}")
