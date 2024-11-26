import unittest
import torch
import numpy as np
from swaglib.optim import sgfs  

class TestSGFS(unittest.TestCase):

    def setUp(self):
        self.lr = 1.0e-5
        self.B = np.random.rand(10, 10)  
        self.model = torch.nn.Linear(10, 1) 
        self.params = {
            'lr': self.lr,
            'B': torch.tensor(self.B, dtype=torch.float32),
            'batch_size': 32,
            'train_size': 100,
            'prec_lik': torch.tensor(1.0),
            'prec_prior': torch.tensor(1.0),
            'gc_norm': 0.1
        }
        self.optimizer = SGFS(self.model.parameters(), **self.params)
        
    def test_initialization(self):
        self.assertEqual(self.optimizer.params['lr'], self.lr)
        self.assertTrue(torch.equal(self.optimizer.params['B'], torch.tensor(self.B, dtype=torch.float32)))

    def test_create_auxiliary_variables(self):
        self.optimizer._create_auxiliary_variables()
        self.assertEqual(self.optimizer.lr.item(), self.lr)
        self.assertEqual(self.optimizer.n_weights, 10)  
        self.assertEqual(self.optimizer.I_t.shape, (self.optimizer.n_weights, self.optimizer.n_weights))
        self.assertEqual(self.optimizer.it.item(), 1.0)

    def test_get_updates(self):
        self.optimizer._create_auxiliary_variables()
        updates, log_likelihood = self.optimizer._get_updates()
        

        self.assertIsInstance(updates, list)
        self.assertGreater(len(updates), 0)


        self.assertIsInstance(log_likelihood, torch.Tensor)

    def test_update_parameters(self):
        initial_weights = [p.clone() for p in self.optimizer.weights]
        self.optimizer._create_auxiliary_variables()
        updates, _ = self.optimizer._get_updates()

        for param, update in updates:
            if param is not self.optimizer.I_t and param is not self.optimizer.it:
                param.data += update  


        for i, param in enumerate(self.optimizer.weights):
            self.assertFalse(torch.equal(param.data, initial_weights[i].data))

if __name__ == '__main__':
    unittest.main()
