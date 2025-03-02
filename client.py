import copy
import torch
import torch.optim as optim

from utils import label_to_onehot, cross_entropy_for_onehot as criterion

class Client:
    def __init__(self, id, model, train_loader, device, lr=0.01, local_epochs=1):
        self.id = id
        self.device = device
        self.model = copy.deepcopy(model).to(device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr)
        self.train_loader = train_loader
        self.local_epochs = local_epochs

    def train(self):
        self.model.train()
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

                # 记录当前batch训练完成后的梯度
                gradients = self.copy_gradients(self.model)
                local_gradients.append(gradients)

                self.optimizer.step()

        return local_gradients, trained_weights

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
