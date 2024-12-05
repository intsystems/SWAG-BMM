import torch
import torch.linalg as linalg
import numpy as np

class SGFS(torch.optim.SGD):

    def __init__(self, initial_lr=1.0e-5, B=None, **kwargs):
        super(SGFS, self).__init__(**kwargs)
        self.params['lr'] = initial_lr
        if B is not None:
            self.params['B'] = torch.tensor(B, dtype=torch.float32)

    def _create_auxiliary_variables(self):
        self.lr = torch.tensor(self.params['lr'], dtype=torch.float32)
        self.n_weights = sum(np.prod(p.shape) for p in self.weights)
        self.I_t = torch.zeros((self.n_weights, self.n_weights), dtype=torch.float32)
        self.it = torch.tensor(1.0)

    def _get_updates(self):
        n = self.params['batch_size']
        N = self.params['train_size']
        prec_lik = self.params['prec_lik']
        prec_prior = self.params['prec_prior']
        gc_norm = self.params['gc_norm']
        gamma = float(n + N) / n

        # compute log-likelihood
        error = self.model_outputs - self.true_outputs
        logliks = log_normal(error, prec_lik)
        sumloglik = logliks.sum()

        # compute gradient of likelihood wrt each data point
        grads = torch.autograd.functional.jacobian(lambda weights: log_normal(self.model_outputs - self.true_outputs, prec_lik, weights), self.weights)
        grads = torch.cat([g.view(-1) for g in grads], dim=0)
        avg_grads = grads.mean(dim=0)
        dist_grads = grads - avg_grads

        # compute variance of gradient
        var_grads = (1. / (n - 1)) * (dist_grads.T @ dist_grads)

        logprior = log_prior_normal(self.weights, prec_prior)
        grads_prior = torch.autograd.grad(logprior, self.weights, create_graph=True)
        grads_prior = torch.cat([g.view(-1) for g in grads_prior])

        # update Fisher information
        I_t_next = (1 - 1 / self.it) * self.I_t + (1 / self.it) * var_grads

        # compute noise
        B = self.params.get('B', gamma * I_t_next * N)
        B_ch = linalg.cholesky(B)
        noise = (2. / torch.sqrt(self.lr)) * B_ch @ torch.randn((self.n_weights, 1))

        # expensive inversion
        inv_cond_mat = gamma * N * I_t_next + (4. / self.lr) * B
        cond_mat = linalg.inv(inv_cond_mat)

        updates = []
        updates.append((self.I_t, I_t_next))
        updates.append((self.it, self.it + 1))

        # update the parameters
        updated_params = 2 * (cond_mat @ (grads_prior + N * avg_grads + noise.flatten()))
        updated_params = updated_params.flatten()
        last_row = 0
        for p in self.weights:
            sub_index = np.prod(p.shape)
            up = updated_params[last_row:last_row + sub_index]
            up = up.view(p.shape)
            updates.append((p, up))
            last_row += sub_index

        return updates, sumloglik

