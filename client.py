import copy
import torch.optim as optim
from opacus import PrivacyEngine

from utils import label_to_onehot, cross_entropy_for_onehot as criterion

class Client:
    def __init__(self, id, model, train_loader, device, lr=0.01, local_epochs=5, sigma=0.3):
        self.id = id
        self.device = device
        self.model = copy.deepcopy(model).to(device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr)
        self.train_loader = train_loader
        self.local_epochs = local_epochs

        self.privacy_engine = PrivacyEngine(accountant="gdp")
        self.model, self.optimizer, self.train_loader = self.privacy_engine.make_private(
            module=self.model,
            optimizer=self.optimizer,
            data_loader=self.train_loader,
            noise_multiplier=sigma,
            max_grad_norm=1.0,
            batch_size=1,
            microbatch_size=1
        )

    def train(self):
        self.model.train()
        local_gradients = []  # 每个batch训练后的梯度
        trained_weights = []  # 每个batch训练前的网络参数
        for epoch in range(self.local_epochs):
            for batch_x, batch_y in self.train_loader:
                if batch_x.size(0) == 0:
                    continue  # 跳过空batch（Opacus可能产生）
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

                self.optimizer.zero_grad()

                outputs = self.model(batch_x)
                loss = criterion(outputs, label_to_onehot(batch_y))
                loss.backward()

                # 记录当前batch训练完成后的梯度
                gradients = self.copy_gradients(self.model)
                local_gradients.append(gradients)

                # 记录当前batch开始前的参数（深拷贝）
                trained_weights.append(self.deep_copy_state_dict(self.model))
                self.optimizer.step()

        return local_gradients, trained_weights

    def get_model_params(self):
        return copy.deepcopy(self.model.state_dict())

    def set_model_params(self, params):
        # 如果模型是由PrivacyEngine包装的，解包后再加载
        if hasattr(self.model, '_module'):
            # 移除 'module.' 前缀
            params = {key.replace('_module.', ''): value for key, value in params.items()}
            self.model._module.load_state_dict(params)  # 访问 _module 属性来加载参数
        else:
            self.model.load_state_dict(params)  # 直接加载原始模型的参数

    def deep_copy_state_dict(self, model):
        # 获取模型的 state_dict 并去除 _module 前缀
        state_dict = model.state_dict()

        # 去除 '_module' 前缀
        modified_state_dict = {key.replace('_module.', ''): value for key, value in state_dict.items()}

        return copy.deepcopy(modified_state_dict)

    def get_privacy_spent(self):
        return self.privacy_engine.get_epsilon(delta=1e-5)

    def copy_gradients(self, model):
        gradients = []
        for param in model.parameters():
            if param.grad is not None:
                gradients.append(param.grad.detach().clone())  # 重要：深度复制，detach防止梯度图留存
            else:
                gradients.append(None)
        return gradients
