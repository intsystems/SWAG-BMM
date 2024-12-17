import torch
import torch.nn as nn
import torch.optim as optim
from swaglib.optim.sgfs import SGFS
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Определяем модель
class ComplexModel(nn.Module):
    def __init__(self):
        super(ComplexModel, self).__init__()
        self.layer1 = nn.Linear(1, 10)
        self.layer2 = nn.Linear(10, 5)
        self.output = nn.Linear(5, 1)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.output(x)
        return x


# Генерация случайных данных для обучения
def generate_data(num_points=100):
    x = torch.linspace(-10, 10, num_points).view(-1, 1)  # Входные данные

    # Создаем зависимость для целевых данных, добавляем синусоиды и шум
    y = 0.5 * (x ** 2) - 2 * torch.sin(x) + 3 + torch.normal(0, 1, x.size())

    return x, y



# Основная функция для демонстрации SGFS
def demo_sgfs(batch_size=16):
    # Генерируем данные
    x, y = generate_data()

    # Создаем dataset и dataloader
    dataset = TensorDataset(x, y)  # Создаем TensorDataset из x и y
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)  # Создаем DataLoader

    # Создаем модель и определяем функцию потерь
    model = ComplexModel()
    criterion = nn.MSELoss()  # Среднеквадратичная ошибка


    sample_optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Обучаем модель
    num_epochs = 2000
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
    #print(f'Model parameters: {list(model.parameters())}')



def demo_sgfs2(batch_size):
    # Загрузка датасета a9a
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
                    'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
                    'hours-per-week', 'native-country', 'income']
    data = pd.read_csv(url, header=None, names=column_names, na_values=' ?', skipinitialspace=True)

    # Предобработка данных
    data.dropna(inplace=True)
    data['income'] = data['income'].apply(lambda x: 1 if x == '>50K' else 0)

    # Преобразование категориальных переменных в числовые
    data = pd.get_dummies(data, drop_first=True)

    # Разделение на признаки и целевую переменную
    X = data.drop('income', axis=1).values
    y = data['income'].values

    # Разделение на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Стандартизация данных
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Преобразование данных в тензоры
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).view(-1, 1)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test).view(-1, 1)

    # Определение модели Байесовой логистической регрессии
    class BayesianLogisticRegression(nn.Module):
        def __init__(self, input_dim):
            super(BayesianLogisticRegression, self).__init__()
            self.linear = nn.Linear(input_dim, 1)

        def forward(self, x):
            return torch.sigmoid(self.linear(x))

    # Инициализация модели, функции потерь и оптимизатора
    input_dim = X_train.shape[1]
    model = BayesianLogisticRegression(input_dim)
    dataset = TensorDataset(X_train_tensor, y_train_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)  # Создаем DataLoader
    # Создаем модель и определяем функцию потерь
    model = BayesianLogisticRegression(input_dim)
    criterion = nn.BCELoss()
    sample_optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Обучаем модель
    num_epochs = 20
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

        # Инициализируем оптимизатор SGFS
        optimizer = SGFS(model.parameters(), lr=0.01, mode="basic", all_gradients=all_gradients)

        # Шаг оптимизации
        optimizer.step()

        # print(all_gradients.keys())

        # Выводим информацию об эпохе
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')



if __name__ == '__main__':
    #demo_sgfs2(batch_size=1024)
    demo_sgfs(batch_size=32)

    #model = ComplexModel()
