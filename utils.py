import numpy as np


def load_dataset():

    with np.load('mnist.npz') as f:

        # Convert RGB to Unit RGB (то есть меняет формат значений от 0 до 1)
        x_train = f['x_train'].astype('float32') / 255
        # reshape from (60000, 28, 28) to (60000, 784)
        x_train = np.reshape(
            x_train, (x_train.shape[0], x_train.shape[1] * x_train.shape[2]))

        # labels
        y_train = f['y_train']
        # convert to output layer format, то есть [1,2,3,4....] -> [[0,1,0,0,0], [0,1,1,0..]...]
        y_train = np.eye(10)[y_train]

        return x_train, y_train
