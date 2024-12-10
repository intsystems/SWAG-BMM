import unittest
import torch
from swaglib.optim.sgdVEM import ConstantSGDasVEM

class TestConstantSGDasVEM(unittest.TestCase):

    def setUp(self):
        # Инициализация параметров и оптимизатора перед каждым тестом
        self.params = [torch.randn(2, 2, requires_grad=True)]  # Параметры для оптимизации
        self.init_lr = 0.01  # Начальная скорость обучения
        self.momentum = 0.9  # Моментум для адаптивной скорости обучения
        self.hyperparameters_list = [torch.tensor(0.1, requires_grad=True)]  # Гиперпараметры
        # Создание экземпляра оптимизатора с заданными параметрами
        self.optimizer = ConstantSGDasVEM(self.params, init_lr=self.init_lr, momentum=self.momentum, hyperparameters_list=self.hyperparameters_list)

    def test_initialization(self):
        # Проверка корректности инициализации оптимизатора
        self.assertEqual(self.optimizer.param_groups[0]["lr"], self.init_lr)  # Проверка скорости обучения
        self.assertEqual(self.optimizer.param_groups[0]["momentum"], self.momentum)  # Проверка моментума
        self.assertTrue(torch.equal(self.optimizer.hypers, torch.tensor(self.hyperparameters_list, requires_grad=True)))  # Проверка гиперпараметров

    def test_step(self):
        # Тестирование метода step оптимизатора
        closure_called = []  # Список для отслеживания вызова closure

        def closure():
            closure_called.append(True)  # Отметка вызова closure
            loss = torch.tensor(1.0, requires_grad=True)  # Пример потерь
            loss.backward()  # Обратное распространение
            return loss

        loss = self.optimizer.step(closure)  # Вызов метода step для обновления параметров
        self.assertTrue(closure_called)  # Проверка, что closure был вызван
        self.assertIsInstance(loss, torch.Tensor)  # Проверка, что возвращаемое значение является тензором

    def test_sgd_function(self):
        # Тестирование функциональности стохастического градиентного спуска
        grads = [torch.randn(2, 2) for _ in self.params]  # Генерация случайных градиентов для параметров
        params = self.params
        momentum_buffer_list = [None for _ in params]  # Инициализация буферов моментума
        self.optimizer.sgd(params, grads, momentum_buffer_list, momentum=self.momentum, lr=self.init_lr, has_sparse_grad=False)  # Вызов метода sgd для обновления параметров

        # Проверка, что обновленные параметры отличаются от исходных
        for i, param in enumerate(params):
            self.assertFalse(torch.equal(param.data, self.params[i].data))  # Убедитесь, что параметры были обновлены

if __name__ == '__main__':
    unittest.main()  # Запуск тестов
