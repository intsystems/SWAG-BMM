import unittest
import torch
from swaglib.optim.sgfs import SGFS
from swaglib.demo.sgfs_demo import SimpleModel, generate_data
import torch.nn as nn

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
        self.assertIsNone(self.optimizer.E)           # Проверяем, что матрица E не инициализирована
        self.assertIsNone(self.optimizer.param_size)   # Проверяем, что размер параметров не установлен

    def test_compute_c(self):
        # Проверка метода compute_c
        gradients_list = [torch.tensor([0.1]), torch.tensor([0.2]), torch.tensor([0.3])]
        C = self.optimizer.compute_c(gradients_list)  # Вычисляем вариацию
        expected_C = torch.tensor(0.01)  # Ожидаемое значение вариации (с учетом векторов)
        self.assertAlmostEqual(C.item(), expected_C.item(), places=6)  # Сравниваем полученное значение с ожидаемым

    def test_compute_h_basic_mode(self):
        # Проверка вычисления H в базовом режиме
        gradients_list = [torch.tensor([0.1]), torch.tensor([0.2]), torch.tensor([0.3])]
        batch_size = len(gradients_list)
        self.optimizer.E = torch.eye(1)  # Устанавливаем единичную матрицу
        C = self.optimizer.compute_c(gradients_list)  # Вычисляем C
        H = self.optimizer.compute_h(C, batch_size)  # Вычисляем H
        self.assertIsInstance(H, torch.Tensor)  # Uбедимся, что H - это тензор
        self.assertEqual(H.size(), (1, 1))  # Uбедимся, что H имеет размер (1, 1)

    def test_step_update(self):
        # Например, используем простую модель
        model = SimpleModel()
        criterion = nn.MSELoss()
        optimizer = SGFS(model.parameters())

        # Генерация входных данных
        x, y = generate_data()

        def closure():
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            return loss

        # Выполнение шага оптимизации
        loss = optimizer.step(closure)  # Теперь должен работать

    def test_invalid_step(self):
        # Проверка поведения метода step при отсутствии градиентов
        self.optimizer.gradients_list = []  # Нет градиентов
        with self.assertRaises(ValueError) as context:
            self.optimizer.step()  # Выполнение шага должно вызвать ошибку
        self.assertEqual(str(context.exception),
                         "The gradients list is empty.")

if __name__ == '__main__':
    #a = [torch.tensor([0.1]), torch.tensor([0.2]), torch.tensor([0.3])]
    #b = torch.stack(a)
    #print(b)


    unittest.main()
