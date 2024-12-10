import unittest
import torch
from swaglib.optim.sgdVEM import ConstantSGDasVEM 

class TestConstantSGDasVEM(unittest.TestCase):

    def setUp(self):
        self.params = [torch.randn(2, 2, requires_grad=True)]  
        self.init_lr = 0.01
        self.momentum = 0.9
        self.hyperparameters_list = [torch.tensor(0.1, requires_grad=True)]  
        self.optimizer = ConstantSGDasVEM(self.params, init_lr=self.init_lr, momentum=self.momentum, hyperparameters_list=self.hyperparameters_list)

    def test_initialization(self):
        self.assertEqual(self.optimizer.param_groups[0]["lr"], self.init_lr)
        self.assertEqual(self.optimizer.param_groups[0]["momentum"], self.momentum)
        self.assertTrue(torch.equal(self.optimizer.hypers, torch.tensor(self.hyperparameters_list, requires_grad=True)))

    def test_step(self):
        closure_called = []

        def closure():
            closure_called.append(True)
            loss = torch.tensor(1.0, requires_grad=True)  
            loss.backward()  
            return loss

        loss = self.optimizer.step(closure)
        self.assertTrue(closure_called)  
        self.assertIsInstance(loss, torch.Tensor)

    def test_sgd_function(self):
        grads = [torch.randn(2, 2) for _ in self.params]  
        params = self.params
        momentum_buffer_list = [None for _ in params]
        self.optimizer.sgd(params, grads, momentum_buffer_list, momentum=self.momentum, lr=self.init_lr, has_sparse_grad=False)

        
        for i, param in enumerate(params):
            self.assertFalse(torch.equal(param.data, self.params[i].data))

if __name__ == '__main__':
    unittest.main()
