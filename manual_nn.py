import numpy as np
import matplotlib.pyplot as plt
from utils import load_dataset


'''
scheme
x784    x20     x10
o       o
o       o       o
o               o
o       o       o
o       o
'''

images, labels = load_dataset()

# Веса
w_input_to_hidden = np.random.uniform(-0.5, 0.5, (20, 784))
w_hidden_to_output = np.random.uniform(-0.5, 0.5, (10, 20))

b_input_to_hidden = np.zeros((20, 1))
b_hidden_to_output = np.zeros((10, 1))


epochs = 3
e_loss = 0
e_correct = 0
learning_rate = 0.01

for epoch in range(epochs):
    print(f'Epoch №{epoch}')

    for image, label in zip(images, labels):
        image = np.reshape(image, (-1, 1))
        label = np.reshape(label, (-1, 1))

        # Forward Propagation - первый этап обучения - прогнать данные через слои и получить выходные данные
        # @ - перемножение матриц

        hiden_raw = b_input_to_hidden + w_input_to_hidden @ image
        # Нормализуем выходные значения = подгоняем под наши
        # LINEAR | RELU | SIGMOID
        # SIGMOID
        hidden = 1 / (1 + np.exp(-hiden_raw))

        output_raw = b_hidden_to_output + w_hidden_to_output @ hidden
        # Нормализуем выходные значения = подгоняем под наши
        # LINEAR | RELU | SIGMOID
        # SIGMOID
        output = 1 / (1 + np.exp(-output_raw))

        #
        #
        #
        # Loss / Error calculation
        # Mean Squaded Error/MSE
        e_loss += 1 / (len(output)) * np.sum((output - label) ** 2, axis=0)
        e_correct += int(np.argmax(output) == np.argmax(label))

        #
        #
        #
        # Back propagation = обратный расчет результаты => рез-ты скрытого слоя и редактирование весов и биасов под полученные ответы

        # Output layer
        delta_output = output - label

        w_hidden_to_output += -learning_rate * \
            delta_output @ np.transpose(hidden)
        b_hidden_to_output += -learning_rate * delta_output

        # Hidden layer
        delta_hidden = np.transpose(
            w_hidden_to_output) @ delta_output * (hidden * (1 - hidden))

        w_input_to_hidden += -learning_rate * \
            delta_hidden @ np.transpose(image)
        b_input_to_hidden += -learning_rate * delta_hidden

    print(f'Loss: {round((e_loss[0] / images.shape[0]) * 100, 3)}%')
    print(f'Accuracy: {round((e_correct / images.shape[0]) * 100, 3)}%')
    e_loss = 0
    e_correct = 0
