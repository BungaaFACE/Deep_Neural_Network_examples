from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.datasets import fashion_mnist
from keras import utils


(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
# конвертируем данные изображений в плоские данные
x_train = x_train.reshape(60000, 784)
# приводим интенсивность пикселей в диапазон 0-1
# или векторизация
x_train = x_train / 255

y_train = utils.to_categorical(y_train, 10)
print(y_train[0])

# Последовательная модель NN
model = Sequential()
# Входные данные размерностью 784, 700 нейронов скрытого слоя, функция активации - relu
model.add(Dense(units=800, input_dim=784, activation='relu'))

model.add(Dense(10, activation='softmax'))

# Компиляция нейронной сети
# loss - функция ошибки
# optimizer - алгоритм обучения
# metrics - метрика качества обучения сети, accuracy - доля правильных ответов
model.compile(loss="categorical_crossentropy",
              optimizer="adam", metrics=["accuracy"])


# обучение нейронной сети
# batch_size - размер выборки (элементов) - то есть после каждых 200 изображений будут меняться веса
# verbose - параметр, включающий отображение прогресса обучения нейронной сети
model.fit(x_train, y_train,
          batch_size=200,
          epochs=50,
          verbose=1)


# Применение нейросети к изображению
x_test = x_test.reshape(10000, 784)
x_test = x_test / 255

y_test = utils.to_categorical(y_test, 10)
predictions = model.predict(x_test)
print(f'correct answer:\n{y_test[0]}')
print(f'NN answer:\n{predictions[0]}')
