import torch

class Server:
    def __init__(self, model):
        self.global_model = model

    def aggregate(self, client_params_list, client_data_sizes):
        total_data = sum(client_data_sizes)
        state_dict = self.global_model.state_dict()  # 先拿到完整结构

        for key in state_dict.keys():
            weighted_sum = 0.0
            for i in range(len(client_params_list)):
                # 统一去除 'module.' 前缀，确保参数键名一致
                client_params = client_params_list[i]
                if key.startswith('_module.'):
                    key = key[len('_module.'):]  # 去掉前缀
                weighted_sum += client_params[key] * (client_data_sizes[i] / total_data)
            state_dict[key] = weighted_sum

        self.global_model.load_state_dict(state_dict)

    def get_global_model_params(self):
        return self.global_model.state_dict()

    def test(self, test_loader, device):
        self.global_model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                outputs = self.global_model(x)
                _, predicted = torch.max(outputs.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        return correct / total
