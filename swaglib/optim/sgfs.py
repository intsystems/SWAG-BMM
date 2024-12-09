
import torch


class SGFS(torch.optim.Optimizer):

    def __init__(self, params, lr=1.0e-2, h_max=None, mode='basic'):
        self.lr = lr
        self.h_max = h_max
        self.mode = mode
        self.gradients_list = []
        self.E = None
        self.param_size = None
        super().__init__(params, defaults={})

    def compute_c(self):
        gradients_tensor = torch.stack(self.gradients_list)
        C = gradients_tensor.var(dim=0, unbiased=False)

        return C

    def compute_h(self, C):
        H = None
        if self.mode == 'basic':
            H = 2/len(self.gradients_list) * torch.inverse(self.lr**2 * C + self.E @ self.E.T)
        elif self.mode == 'diagonal':
            if self.h_max is not None:
                one_vec = torch.ones(C.size(dim=0))
                h_vec = self.h_max * one_vec
                H = torch.diag(h_vec)
            else:
                B = torch.linalg.cholesky(C)
                B_d = torch.diagonal(B)
                E_d = torch.diagonal(self.E)
                H_vec = 1 / (torch.mul(self.lr**2 * B_d * B_d, E_d * E_d))
                H = 2 / len(self.gradients_list) * torch.diag(H_vec)
        elif self.mode == 'scalar':
            one_vec = torch.ones(C.size(dim=0))
            B = torch.linalg.cholesky(C)
            B_d = torch.diagonal(B)
            E_d = torch.diagonal(self.E)
            H_scalar = 2 * C.size(dim=0) / len(self.gradients_list) / torch.sum(torch.mul(self.lr**2 * B_d * B_d, E_d * E_d))
            H = torch.diag(H_scalar * one_vec)
        return H


    def step(self, closure = None):
        self.gradients_list = []
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    self.gradients_list.append(param.grad.data.clone())
                    self.param_size = param.data.size()

        C = self.compute_c()
        self.E = torch.eye(self.param_size)
        H = self.compute_h(C)
        for group in self.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    noise = torch.distributions.MultivariateNormal(torch.zeros(self.param_size), covariance_matrix=(self.E @ self.E.T))
                    param.data -= self.lr**2 * H @ param.grad - self.lr * (H @ (self.E @ noise.sample()))

        return loss

