import torch

class SGFS(torch.optim.Optimizer):
    def __init__(self, params, lr=1.0e-2, h_max=None, mode='basic'):
        """
        Инициализация оптимизатора SGFS.
        
        :param params: Параметры модели, которые нужно оптимизировать.
        :param lr: learning rate.
        :param h_max: Максимальное значение h, используется в режиме 'diagonal'.
        :param mode: Режим оптимизации ('basic', 'diagonal', 'scalar').
        """
        self.lr = lr  # Скорость обучения
        self.h_max = h_max  # Максимальное значение h
        self.mode = mode  # Режим работы оптимизатора
        self.gradients_list = []  # Список для хранения градиентов
        self.E = None  # Матрица E (для управления шумом)
        self.param_size = None  # Размер параметров
        super().__init__(params, defaults={})  # Инициализация базового класса

    def compute_c(self):
        """
        Вычисляет матрицу C, содержащую дисперсию градиентов.
        
        :return: Дисперсия градиентов.
        """
        gradients_tensor = torch.stack(self.gradients_list)  # Стек градиентов в тензор
        C = gradients_tensor.var(dim=0, unbiased=False)  # Вычисление дисперсии по первому измерению

        return C

    def compute_h(self, C):
        """
        Вычисляет матрицу H в зависимости от выбранного режима.

        :param C: Матрица дисперсии градиентов.
        :return: Матрица H, используемая для обновления параметров.
        """
        H = None  # Инициализация H
        if self.mode == 'basic':
            # Вычисление H для базового режима
            H = 2 / len(self.gradients_list) * torch.inverse(self.lr**2 * C + self.E @ self.E.T)
        elif self.mode == 'diagonal':
            # Вычисление H для диагонального режима
            if self.h_max is not None:
                one_vec = torch.ones(C.size(dim=0))  # Вектор единиц для размера C
                h_vec = self.h_max * one_vec  # Вектор максимальных значений h
                H = torch.diag(h_vec)  # Создание диагональной матрицы H
            else:
                B = torch.linalg.cholesky(C)  # Холоджизование матрицы C
                B_d = torch.diagonal(B)  # Диагональные элементы B
                E_d = torch.diagonal(self.E)  # Диагональные элементы E
                H_vec = 1 / (torch.mul(self.lr**2 * B_d * B_d, E_d * E_d))  # Вычисление вектора H
                H = 2 / len(self.gradients_list) * torch.diag(H_vec)  # Создание диагональной матрицы H
        elif self.mode == 'scalar':
            # Вычисление H для скалярного режима
            one_vec = torch.ones(C.size(dim=0))
            B = torch.linalg.cholesky(C)
            B_d = torch.diagonal(B)
            E_d = torch.diagonal(self.E)
            H_scalar = 2 * C.size(dim=0) / len(self.gradients_list) / torch.sum(torch.mul(self.lr**2 * B_d * B_d, E_d * E_d))
            H = torch.diag(H_scalar * one_vec)  # Создание диагональной матрицы H
        return H

    def step(self, closure=None):
        """
        Выполняет один шаг оптимизации, обновляя параметры.
        
        :param closure: Функция для вычисления потерь (loss) и градиентов.
        :return: Значение потерь после шага.
        """
        self.gradients_list = []  # Очистка списка градиентов
        loss = None  # Инициализация переменной потерь
        if closure is not None:
            loss = closure()  # Вычисление потерь с использованием замыкания

        # Сбор градиентов от всех параметров
        for group in self.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    self.gradients_list.append(param.grad.data.clone())  # Добавление градиента в список
                    self.param_size = param.data.size()  # Запись размера параметров

        C = self.compute_c()  # Вычисление матрицы C из градиентов
        self.E = torch.eye(self.param_size)  # Инициализация матрицы E как единичной матрицы
        H = self.compute_h(C)  # Вычисление матрицы H

        # Обновление параметров модели
        for group in self.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    # Генерация шума с использованием многомерного нормального распределения
                    noise = torch.distributions.MultivariateNormal(
                        torch.zeros(self.param_size), 
                        covariance_matrix=(self.E @ self.E.T)
                    )
                    # Обновление параметров с учетом градиентов и шума
                    param.data -= self.lr**2 * H @ param.grad - self.lr * (H @ (self.E @ noise.sample()))

        return loss  # Возвращение значения потерь
