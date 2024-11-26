import unittest
import torch
from swaglib.optim import SGDEntropyEstimator  

class TestSGDEntropyEstimator(unittest.TestCase):

    def setUp(self):
        self.lr = 0.01
        self.sigma_0 = 1.0
        self.N_iter = 100
        self.params = [torch.randn(2, 2, requires_grad=True)]  
        self.optimizer = SGDEntropyEstimator(self.params, N_iter=self.N_iter, lr=self.lr, sigma_0=self.sigma_0)

    def test_initialization(self):
        self.assertEqual(self.optimizer.lr, self.lr)
        self.assertEqual(self.optimizer.sigma_0, self.sigma_0)
        self.assertEqual(self.optimizer.N_iter, self.N_iter)
        self.assertIsInstance(self.optimizer.rs, torch.Generator)

    def test_entropy_initialization(self):
        D = sum(p.numel() for p in self.params)
        expected_entropy = 0.5 * D * (1 + torch.log(torch.tensor(2 * torch.pi))) + D * torch.log(torch.tensor(self.sigma_0))
        self.assertAlmostEqual(self.optimizer.entropy.item(), expected_entropy.item(), places=6)

    def test_step_with_closure(self):
        closure_called = []

        def closure():
            closure_called.append(True)
            loss = torch.tensor(1.0, requires_grad=True)  
            loss.backward()  # Учет градиента
            return loss

        loss, entropy = self.optimizer.step(closure)
        self.assertTrue(closure_called)  
        self.assertIsInstance(loss, torch.Tensor)
        self.assertIsInstance(entropy, float)

    def test_approx_log_det(self):
        grads = [torch.randn(2, 2) for _ in self.params]  
        approx_entropy = self.optimizer._approx_log_det(grads)
        self.assertIsInstance(approx_entropy, float)

    def test_exact_log_det(self):
        grads = [torch.randn(2, 2) for _ in self.params] 
        exact_entropy = self.optimizer._exact_log_det(grads)
        self.assertIsInstance(exact_entropy, float)

    def test_jvp(self):
        grads = [torch.randn(2, 2) for _ in self.params]  
        vector = torch.randn(4)  # Вектор для умножения
        jvp_result = self.optimizer._jvp(grads, vector)
        self.assertIsInstance(jvp_result, torch.Tensor)
        self.assertEqual(jvp_result.shape, vector.shape)

if __name__ == '__main__':
    unittest.main()
