import torch


class SGFS(torch.optim.Optimizer):
    def __init__(self, params, lr=1.0e-2, h_max=None, mode='basic', all_gradients=None):
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
        self.all_gradients = all_gradients  # Словарь с градиентами
        self.E = None  # Матрица E (для управления шумом)

        super().__init__(params, defaults={})  # Инициализация базового класса

    def compute_c(self, gradients_list):
        """
        Вычисляет матрицу C, содержащую дисперсию градиентов.

        :return: Дисперсия градиентов.
        """
        d = len(gradients_list)
        prod_shape = torch.tensor(gradients_list[0].shape).prod().item()
        gradients_tensor = torch.stack(gradients_list)
        flattened_tensor = gradients_tensor.view(d, prod_shape)
        covariance_matrix = torch.cov(flattened_tensor.T).view(prod_shape, prod_shape)

        return covariance_matrix

    def compute_h(self, C, batch_size):
        """
        Вычисляет матрицу H в зависимости от выбранного режима.

        :param C: Матрица дисперсии градиентов.
        :return: Матрица H, используемая для обновления параметров.
        """
        H = None  # Инициализация H
        if self.mode == 'basic':
            # Вычисление H для базового режима
            H = 2 / batch_size * torch.inverse(self.lr ** 2 * C + self.E @ self.E.T)
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
                H_vec = 1 / (torch.mul(self.lr ** 2 * B_d * B_d, E_d * E_d))  # Вычисление вектора H
                H = 2 / batch_size * torch.diag(H_vec)  # Создание диагональной матрицы H
        elif self.mode == 'scalar':
            # Вычисление H для скалярного режима
            one_vec = torch.ones(C.size(dim=0))
            B = torch.linalg.cholesky(C)
            B_d = torch.diagonal(B)
            E_d = torch.diagonal(self.E)
            H_scalar = 2 * C.size(dim=0) / batch_size / torch.sum(
                torch.mul(self.lr ** 2 * B_d * B_d, E_d * E_d))
            H = torch.diag(H_scalar * one_vec)  # Создание диагональной матрицы H
        return H


    def step(self, closure=None):
        """
        Выполняет один шаг оптимизации, обновляя параметры.

        :param closure: Функция для вычисления потерь (loss) и градиентов.
        :return: Значение потерь после шага.
        """
        if self.all_gradients == {}:
            raise ValueError("The gradients list is empty.")
        loss = None  # Инициализация переменной потерь
        if closure is not None:
            loss = closure()  # Вычисление потерь с использованием замыкания

        # Цикл по параметрам
        for group in self.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    gradients_list = self.all_gradients[param]
                    shape = gradients_list[0].shape
                    prod_shape = torch.tensor(gradients_list[0].shape).prod().item()
                    grad = torch.mean(torch.stack(gradients_list)).view(prod_shape).unsqueeze(-1)
                    C = self.compute_c(gradients_list) # Вычисление матрицы(!) C из градиентов
                    batch_size = len(gradients_list)  # размер батча
                    if not gradients_list:
                        raise ValueError("The gradients list is empty.")

                    self.E = torch.eye(prod_shape)  # Инициализация матрицы E как единичной матрицы(!)
                    H = self.compute_h(C, batch_size)  # Вычисление матрицы(!) H
                    noise = torch.distributions.MultivariateNormal(
                        torch.zeros(param.size()),
                        covariance_matrix=(self.E @ self.E.T)
                    )
                    # Обновление параметров с учетом градиентов и шума
                    vector_delta = self.lr ** 2 * H @ grad - self.lr * (H @ (self.E @ noise.sample()))
                    tensor_delta = vector_delta.view(*shape)
                    param.data -= tensor_delta

        return loss  # Возвращение значения потерь
