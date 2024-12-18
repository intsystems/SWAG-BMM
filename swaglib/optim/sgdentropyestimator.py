import torch

class SGDEntropyEstimator(torch.optim.SGD):
    def __init__(self, params, N_iter=100, lr=0.01, sigma_0=1.0, is_approx=True, rs=None, callback=None, **kwargs):
        super().__init__(params, lr=lr, **kwargs)
        self.sigma_0 = sigma_0
        self.N_iter = N_iter
        self.lr = lr
        self.approx = is_approx
        self.rs = rs if rs is not None else torch.Generator().manual_seed(0)
        self.callback = callback

        D = sum(p.numel() for p in self.param_groups[0]['params'])
        self.entropy = 0.5 * D * (1 + torch.log(torch.tensor(2 * torch.pi))) + D * torch.log(torch.tensor(self.sigma_0))

    def step(self, closure=None):
        for t in range(self.N_iter):
            loss = None
            if closure is not None:
                loss = closure()

            super().step(closure)

            with torch.no_grad():
                grads = [p.grad for p in self.param_groups[0]['params']]
                if self.approx:
                    self.entropy += self._approx_log_det(grads)
                else:
                    self.entropy += self._exact_log_det(grads)

                if self.callback:
                    current_params = [p.clone() for p in self.param_groups[0]['params']]
                    self.callback(x=current_params, t=t, entropy=self.entropy)
                print(self.entropy)

        return loss, self.entropy

    def _approx_log_det(self, grads):
        D = sum(g.numel() for g in grads)
        R0 = torch.randn(D, generator=self.rs, device=grads[0].device)
        R1 = self._jvp(grads, R0)
        R2 = self._jvp(grads, R1)
        return torch.dot(R0, -2 * R0 + 3 * R1 - R2).item()

    def _exact_log_det(self, grads):
        D = sum(g.numel() for g in grads)
        mat = torch.zeros(D, D, device=grads[0].device)
        eye = torch.eye(D, device=grads[0].device)

        for i in range(D):
            mat[:, i] = self._jvp(grads, eye[:, i])

        return torch.logdet(mat).item()

    def _jvp(self, grads, vector):
        flat_grads = torch.cat([g.view(-1) for g in grads])
        return vector - self.lr * flat_grads * vector
