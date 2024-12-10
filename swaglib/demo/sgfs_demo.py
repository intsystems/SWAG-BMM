import torch
import torch.nn as nn
import torch.optim as optim
from swaglib.optim.sgfs import SGFS
from torch.utils.data import DataLoader, TensorDataset


# Определяем простую линейную модель
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(1, 1)  # Линейный слой

    def forward(self, x):
        return self.linear(x)  # Прямой проход через модель


# Генерация случайных данных для обучения
def generate_data(num_points=100):
    x = torch.linspace(-10, 10, num_points).view(-1, 1)  # Входные данные
    y = 2 * x + 3 + torch.normal(0, 1, x.size())  # Целевые данные с шумом
    return x, y



# Основная функция для демонстрации SGFS
def demo_sgfs(batch_size=16):
    # Генерируем данные
    x, y = generate_data()

    # Создаем dataset и dataloader
    dataset = TensorDataset(x, y)  # Создаем TensorDataset из x и y
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)  # Создаем DataLoader

    # Создаем модель и определяем функцию потерь
    model = SimpleModel()
    criterion = nn.MSELoss()  # Среднеквадратичная ошибка


    sample_optimizer = optim.SGD(model.parameters(), lr=0.01)




    # Обучаем модель
    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()  # Устанавливаем модель в режим обучения

        # Словарь для накопления градиентов по всем батчам
        all_gradients = {param: [] for param in model.parameters()}

        # Проходим по всем батчам
        for batch_x, batch_y in dataloader:

            # Обнуляем градиенты перед каждой итерацией
            sample_optimizer.zero_grad()

            # Прямой проход
            outputs = model(batch_x)

            # Вычисление потерь
            loss = criterion(outputs, batch_y)

            # Обратный проход
            loss.backward()

            gradients_dict = {param: param.grad.detach().cpu() for param in model.parameters()}
            for name in all_gradients.keys():
                all_gradients[name].append(gradients_dict[name])



        #Инициализируем оптимизатор SGFS
        optimizer = SGFS(model.parameters(), lr=0.01, all_gradients=all_gradients)

        #Шаг оптимизации
        optimizer.step()

        #print(all_gradients.keys())

        # Выводим информацию об эпохе
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Выводим параметры после обучения
    print(f'Model parameters: {list(model.parameters())}')


if __name__ == '__main__':
    demo_sgfs(batch_size=16)

    #model = SimpleModel()

    #for name, param in model.named_parameters():
        #print(f"Parameter: {name}, Size: {param.size()}")
