import copy
import torch
import torch.optim as optim

from utils import label_to_onehot, cross_entropy_for_onehot as criterion

class Client:
    def __init__(self, id, model, train_loader, device, lr=0.01, local_epochs=1, sigma=1e-5):
        self.id = id
        self.device = device
        self.model = copy.deepcopy(model).to(device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr)
        self.train_loader = train_loader
        self.local_epochs = local_epochs
        self.sigma = sigma  # 高斯噪声参数
        self.max_grad_norm = 1.0  # 梯度裁剪阈值

    def train(self):
        self.model.train()
        real_images = []
        local_gradients = []  # 每个batch训练后的梯度
        trained_weights = []  # 每个batch训练前的网络参数
        for epoch in range(self.local_epochs):
            for batch_x, batch_y in self.train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

                # 记录当前batch开始前的参数（深拷贝）
                trained_weights.append(self.deep_copy_state_dict(self.model))

                self.optimizer.zero_grad()

                outputs = self.model(batch_x)
                loss = criterion(outputs, label_to_onehot(batch_y))
                loss.backward()

                real_images.append(batch_x.detach().cpu())

                # # 记录当前batch训练完成后的梯度
                gradients = self.copy_gradients(self.model)
                # local_gradients.append(gradients)

                # 对关键梯度加噪
                noisy_gradients = self.add_topk_noise(gradients)

                # 更新到模型
                self.apply_noisy_gradients(noisy_gradients)

                # 保存加噪后的梯度，用于DLG
                local_gradients.append(noisy_gradients)

                self.optimizer.step()

        return local_gradients, trained_weights,real_images

    def get_model_params(self):
        return copy.deepcopy(self.model.state_dict())

    def set_model_params(self, params):
        self.model.load_state_dict(params)

    def deep_copy_state_dict(self, model):
        return copy.deepcopy(model.state_dict())

    def copy_gradients(self, model):
        gradients = []
        for param in model.parameters():
            if param.grad is not None:
                gradients.append(param.grad.detach().clone())  # 重要：深度复制，detach防止梯度图留存
            else:
                gradients.append(None)
        return gradients

    def add_topk_noise(self, gradients):
        """
        对每个参数的L2范数进行排序，Top-k加高斯噪声，其他保留原始裁剪后梯度。
        """
        # # === 先做全局裁剪 ===
        # global_grad_vec = torch.cat([g.view(-1) for g in gradients if g is not None])
        # global_norm = torch.norm(global_grad_vec, p=2)
        # clip_coef = self.max_grad_norm / (global_norm + 1e-6)
        # if clip_coef < 1.0:
        #     for g in gradients:
        #         if g is not None:
        #             g.mul_(clip_coef)

        # === 逐参数L2范数统计 ===
        grad_norms = []
        grad_shapes = []
        for g in gradients:
            if g is None:
                grad_norms.append(None)
                grad_shapes.append(None)
                continue
            if g.dim() <= 1:
                norms = g.abs()
            else:
                norms = g.view(g.shape[0], -1).norm(dim=1)

            grad_norms.append(norms)
            grad_shapes.append(g.shape)

        # === Flatten后做Top-k筛选 ===
        flat_norms = torch.cat([n.view(-1) for n in grad_norms if n is not None])
        k = max(1, int(len(flat_norms) * 0.5))  # 调整加噪比例
        threshold = torch.topk(flat_norms, k).values.min()

        # === 遍历每层逐参数加噪 ===
        noisy_gradients = []
        for i, g in enumerate(gradients):
            if g is None:
                noisy_gradients.append(None)
                continue

            norms = grad_norms[i]
            shape = grad_shapes[i]

            if norms is not None:
                if norms.dim() == 0:
                    mask = (norms >= threshold).float()
                    noise = torch.randn_like(g) * self.sigma
                    g_noisy = g + mask * noise
                else:
                    mask = (norms >= threshold).float().view([-1] + [1] * (g.dim() - 1))
                    noise = torch.randn_like(g) * self.sigma
                    g_noisy = g + mask * noise

                noisy_gradients.append(g_noisy)

        return noisy_gradients

    def apply_noisy_gradients(self, noisy_gradients):
        with torch.no_grad():
            for param, noisy_grad in zip(self.model.parameters(), noisy_gradients):
                if noisy_grad is not None:
                    param.grad = noisy_grad.clone()
