import unittest
import torch
from swaglib.optim.sgdentropyestimator import SGDEntropyEstimator

class TestSGDEntropyEstimator(unittest.TestCase):

    def setUp(self):
        # Настройка параметров для оптимизатора перед каждым тестом
        self.lr = 0.01
        self.sigma_0 = 1.0
        self.N_iter = 100
        # Инициализация случайных параметров с градиентами
        self.params = [torch.randn(2, 2, requires_grad=True)]
        # Создание экземпляра оптимизатора SGDEntropyEstimator
        self.optimizer = SGDEntropyEstimator(self.params, N_iter=self.N_iter, lr=self.lr, sigma_0=self.sigma_0)

    def test_initialization(self):
        # Проверка инициализации параметров оптимизатора
        self.assertEqual(self.optimizer.lr, self.lr)  # ПроверкаLearning rate
        self.assertEqual(self.optimizer.sigma_0, self.sigma_0)  # Проверка начального значения σ
        self.assertEqual(self.optimizer.N_iter, self.N_iter)  # Проверка числа итераций
        self.assertIsInstance(self.optimizer.rs, torch.Generator)  # Проверка генератора случайных чисел

    def test_entropy_initialization(self):
        # Проверка корректности инициализации энтропии
        D = sum(p.numel() for p in self.params)  # Определение общего числа параметров
        expected_entropy = 0.5 * D * (1 + torch.log(torch.tensor(2 * torch.pi))) + D * torch.log(torch.tensor(self.sigma_0))
        # Сравнение с ожидаемым значением энтропии
        self.assertAlmostEqual(self.optimizer.entropy.item(), expected_entropy.item(), places=6)

    def test_step_with_closure(self):
        # Тестирование метода step с замыканием (closure)
        closure_called = []  # Список для отслеживания вызова замыкания

        def closure():
            closure_called.append(True)  # Отображение вызова замыкания
            loss = torch.tensor(1.0, requires_grad=True)  # Создание тензора потерь
            loss.backward()  # Учет градиента
            return loss

        # Выполнение шага оптимизации
        loss, entropy = self.optimizer.step(closure)
        # Проверка, что замыкание было вызвано
        self.assertTrue(closure_called)
        self.assertIsInstance(loss, torch.Tensor)  # Проверка, что loss - тензор
        self.assertIsInstance(entropy, float)  # Проверка, что энтропия - число с плавающей запятой

    def test_approx_log_det(self):
        # Проверка функции приближенной логарифмической детерминанты
        grads = [torch.randn(2, 2) for _ in self.params]  # Генерация случайных градиентов
        approx_entropy = self.optimizer._approx_log_det(grads)  # Вычисление приближенной энтропии
        self.assertIsInstance(approx_entropy, float)  # Проверка, что результат - число

    def test_exact_log_det(self):
        # Проверка функции точной логарифмической детерминанты
        grads = [torch.randn(2, 2) for _ in self.params]  # Генерация случайных градиентов
        exact_entropy = self.optimizer._exact_log_det(grads)  # Вычисление точной энтропии
        self.assertIsInstance(exact_entropy, float)  # Проверка, что результат - число

    def test_jvp(self):
        # Проверка функции производной по направлению (JVP)
        grads = [torch.randn(2, 2) for _ in self.params]  # Генерация случайных градиентов
        vector = torch.randn(4)  # Генерация случайного вектора
        jvp_result = self.optimizer._jvp(grads, vector)  # Вычисление JVP
        self.assertIsInstance(jvp_result, torch.Tensor)  # Проверка, что результат - тензор
        self.assertEqual(jvp_result.shape, vector.shape)  # Проверка совместимости размерностей с направляющим вектором

# Запуск тестов, если данный файл исполняемый
if __name__ == '__main__':
    unittest.main()
