import unittest
import torch
from swaglib.optim.sgld import SGLD

class TestSGLD(unittest.TestCase):

    def setUp(self):
        # Устанавливаем параметры для тестов
        self.params = [torch.tensor([1.0], requires_grad=True)]
        self.lr_param = 1.0
        self.prior_param = 1.0
        self.scaler = 1.0
        self.m_scaler = 1

    def test_initialization_default(self):
        # Проверяем инициализацию с использованием значений по умолчанию
        optimizer = SGLD(self.params)
        self.assertEqual(optimizer.lr_mode, "a/t")
        self.assertEqual(optimizer.lr_param, 1)
        self.assertEqual(optimizer.prior, "gauss")

    def test_initialization_custom(self):
        # Проверяем инициализацию с пользовательскими значениями
        optimizer = SGLD(
            self.params,
            lr_mode="a/t^b",
            lr_param=(1.0, 2.0),
            prior="laplace",
            prior_param=self.prior_param,
            scaler=self.scaler,
            m_scaler=self.m_scaler
        )
        self.assertEqual(optimizer.lr_mode, "a/t^b")
        self.assertEqual(optimizer.prior, "laplace")
        self.assertEqual(optimizer.lr_param, (1.0, 2.0))

    def test_initialization_invalid_lr_mode(self):
        # Проверяем инициализацию с неправильным lr_mode
        with self.assertRaises(ValueError):
            SGLD(self.params, lr_mode="invalid_mode")

    def test_initialization_invalid_prior(self):
        # Проверяем инициализацию с неправильным prior
        with self.assertRaises(ValueError):
            SGLD(self.params, prior="invalid_prior")


    def test_step_size_a_t(self):
        optimizer = SGLD(self.params, lr_param=self.lr_param, scaler=self.scaler, m_scaler=self.m_scaler)
        optimizer.iter = 1  # Устанавливаем итерацию
        self.assertAlmostEqual(optimizer.step_size(), 1.0)

    def test_prior_gradient_gauss(self):
        optimizer = SGLD(self.params, prior="gauss", prior_param=self.prior_param, scaler=self.scaler, m_scaler=self.m_scaler)
        grad = optimizer.prior_gradient(torch.tensor([0.5]))
        self.assertEqual(grad.item(), -0.5)  # Проверяем, что градиент корректен

    def test_prior_gradient_laplace(self):
        optimizer = SGLD(self.params, prior="laplace", prior_param=self.prior_param, scaler=self.scaler, m_scaler=self.m_scaler)
        grad = optimizer.prior_gradient(torch.tensor([2.0]))
        self.assertEqual(grad.item(), -1.0)  # Проверяем, что градиент корректен

    def test_step(self):
        # Проверяем метод step()
        optimizer = SGLD(self.params, lr_param=self.lr_param, scaler=self.scaler, m_scaler=self.m_scaler)
        self.params[0].grad = torch.tensor([1.0])  # Устанавливаем градиент
        optimizer.step()  # Выполняем шаг
        # Проверяем, что параметр изменился
        self.assertNotEqual(self.params[0].item(), 1.0)


if __name__ == '__main__':
    unittest.main()
