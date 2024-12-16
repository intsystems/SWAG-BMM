import torch


class SGLD(torch.optim.Optimizer):
    def __init__(self, params, lr_mode=None, lr_param=None, prior=None, prior_param=None, scaler=None, m_scaler=None):
        if lr_mode is None:
            self.lr_mode = "a/t"
            self.lr_param = 1
        elif lr_mode in ["a/t", "a/t^b", "a(b+t)^-g"]:
            self.lr_mode = lr_mode
            self.lr_param = lr_param
        else:
            raise ValueError("Wrong lr_mode")

        if prior is None:
            self.prior = "gauss"
            self.prior_param = 1
        elif prior in ["gauss", "laplace"]:
            self.prior = prior
            self.prior_param = prior_param
        else:
            raise ValueError("Wrong prior")
        if scaler is None:
            self.scaler = 1
        else:
            self.scaler = scaler

        if m_scaler is None:
            self.m_scaler = 1
        else:
            self.m_scaler = m_scaler

        self.iter = 0
        super().__init__(params, defaults={})  # Инициализация базового класса

    def step_size(self):
        if self.lr_mode == "a/t":
            return self.lr_param / self.iter
        elif self.lr_mode == "1/t^b":
            return self.lr_param[0] / pow(self.iter, self.lr_param[1])
        else:
            return self.lr_param[0] / pow(self.lr_param[1] + self.iter, self.lr_param[2])

    def prior_gradient(self, data):
        if self.prior == "laplace":
            return - torch.sign(data) / self.prior_param
        elif self.prior == "gauss":
            return -data / self.prior_param

    def step(self, closure=None):
        self.iter += 1
        loss = None  # Инициализация переменной потерь
        if closure is not None:
            loss = closure()  # Вычисление потерь с использованием замыкания

        # Цикл по параметрам
        for group in self.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    lr = self.step_size()
                    shape = param.grad.shape
                    prod_shape = torch.tensor(param.grad.shape).prod().item()
                    grad = param.grad.view(prod_shape).unsqueeze(-1)
                    M = self.m_scaler * torch.eye(grad.size()[0])
                    noise = torch.distributions.MultivariateNormal(
                        torch.zeros(grad.size()[0]),
                        covariance_matrix=(lr * M)
                    )
                    vector_delta = lr / 2 * (M @ (self.prior_gradient(param.data.view(prod_shape).unsqueeze(-1)) +
                                                        self.scaler * grad)) + noise.sample().unsqueeze(-1)
                    tensor_delta = vector_delta.view(*shape)
                    param.data -= tensor_delta

        return loss
