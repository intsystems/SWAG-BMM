import unittest
import torch
from swaglib.optim.sgfs import SGFS

class TestSGFS(unittest.TestCase):

    def setUp(self):
        # Устанавливаем параметры для оптимизатора перед каждым тестом
        self.lr = 0.01         # Исходный learning rate
        self.h_max = None      # Максимальное значение h
        self.mode = 'basic'    # Режим работы оптимизатора
        # Создаем тензор параметров с градиентами
        self.params = [torch.randn(2, 2, requires_grad=True)]
        # Инициализируем оптимизатор SGFS с заданными параметрами
        self.optimizer = SGFS(self.params, lr=self.lr, h_max=self.h_max, mode=self.mode)

    def test_initialization(self):
        # Проверка инициализации параметров оптимизатора
        self.assertEqual(self.optimizer.lr, self.lr)  # Сравниваем learning rate
        self.assertIsNone(self.optimizer.h_max)       # Проверяем, что h_max None
        self.assertEqual(self.optimizer.mode, self.mode)  # Сравниваем режим
        self.assertEqual(self.optimizer.gradients_list, [])  # Проверяем, что список градиентов пуст
        self.assertIsNone(self.optimizer.E)           # Проверяем, что матрица E не инициализирована
        self.assertIsNone(self.optimizer.param_size)   # Проверяем, что размер параметров не установлен

    def test_compute_c(self):
        # Проверка метода compute_c
        self.optimizer.gradients_list = [torch.tensor([0.1]), torch.tensor([0.2]), torch.tensor([0.3])]
        C = self.optimizer.compute_c()  # Вычисляем вариацию
        expected_C = torch.tensor(0.02)  # Ожидаемое значение вариации (с учетом векторов)
        self.assertAlmostEqual(C.item(), expected_C.item(), places=6)  # Сравниваем полученное значение с ожидаемым

    def test_compute_h_basic_mode(self):
        # Проверка вычисления H в базовом режиме
        self.optimizer.gradients_list = [torch.tensor([0.1]), torch.tensor([0.2]), torch.tensor([0.3])]
        self.optimizer.E = torch.eye(1)  # Устанавливаем единичную матрицу
        C = self.optimizer.compute_c()  # Вычисляем C
        H = self.optimizer.compute_h(C)  # Вычисляем H
        self.assertIsInstance(H, torch.Tensor)  # Uбедимся, что H - это тензор
        self.assertEqual(H.size(), (1, 1))  # Uбедимся, что H имеет размер (1, 1)

    def test_step_update(self):
        # Проверка выполнения шага обновления параметров
        def closure():
            # Замыкание для вычисления потерь
            loss = torch.tensor(1.0, requires_grad=True)
            loss.backward()  # Учет градиентов
            return loss

        self.optimizer.step(closure)  # Выполнение шага оптимизации

        # Проверяем, что параметры были обновлены путем сравнения их значений
        for param in self.params:
            self.assertFalse(
                torch.allclose(param.data, torch.zeros_like(param.data)))  # Проверяем, что параметры изменились

    def test_invalid_step(self):
        # Проверка поведения метода step при отсутствии градиентов
        self.optimizer.gradients_list = []  # Нет градиентов
        loss = self.optimizer.step()  # Выполнение шага
        self.assertIsNone(loss)  # Убедитесь, что loss остается None, как и ожидалось

if __name__ == '__main__':
    unittest.main()
