import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer
from typing import List, Optional

class ConstantSGDasVEM(Optimizer):
    def __init__(self, params, init_lr=0.01, momentum=0.0, hyperparameters_list = None, lr_est_lr = 0.01):

        self.hypers = torch.tensor(hyperparameters_list, requires_grad=True)

        defaults = dict(lr = init_lr, momentum=momentum, hyperparameters_list = hyperparameters_list, lr_est_lr = lr_est_lr)
        super(ConstantSGDasVEM, self).__init__(params, defaults)


    def _init_group(self, group, params, grads, momentum_buffer_list, hypers, hypers_grads):
        has_sparse_grad = False

        for p in group["params"]:
            if p.grad is not None:
                params.append(p)
                grads.append(p.grad)
                if p.grad.is_sparse:
                    has_sparse_grad = False

                if group["momentum"] != 0:
                    state = self.state[p]
                    momentum_buffer_list.append(state.get("momentum_buffer"))

        for p in group['hyperparameters_list']:
          if p.grad is not None:
            hypers.append(p)
            hypers_grads.append(p.grad)

        return has_sparse_grad

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params: List[Tensor] = []
            grads: List[Tensor] = []
            momentum_buffer_list: List[Optional[Tensor]] = []
            hypers: List[Tensor] = []
            hypers_grads = List[Tensor] = []

            has_sparse_grad = self._init_group(
                group, params, grads, momentum_buffer_list, hypers, hypers_grads
            )

            sgd(
                params,
                grads,
                momentum_buffer_list,
                momentum=group["momentum"],
                lr=group["lr"],
                has_sparse_grad=has_sparse_grad,
            )

            if group["momentum"] != 0:
                for p, momentum_buffer in zip(params, momentum_buffer_list):
                    state = self.state[p]
                    state["momentum_buffer"] = momentum_buffer

        group['lr'] = get_current_lr_est(group['lr_est_lr'])

        for p in group['hyperparameters_list']:
            hypers: List[Tensor] = []
            hypers_grads = List[Tensor] = []

        return loss

    def sgd(
        params: List[Tensor],
        grads: List[Tensor],
        momentum_buffer_list: List[Optional[Tensor]],
        *,
        momentum: float,
        lr: float,
        has_sparse_grad: bool,
    ):

        for i, param in enumerate(params):
            grad = grads[i]

            if momentum != 0:
                buf = momentum_buffer_list[i]

                if buf is None:
                    buf = torch.clone(grad).detach()
                    momentum_buffer_list[i] = buf

                grad = buf

            param.add_(grad, alpha=-lr)

    def get_current_lr_est(lr, init_lr):
      return init_lr