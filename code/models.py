import tensorflow as tf
from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense

import hyperparameters as hp

class YourModel(tf.keras.Model):
    def __init__(self):
        super(YourModel, self).__init__()

        self.optimizer = tf.keras.optimizers.Adam()

        self.architecture = [
            Conv2D(96, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(hp.img_size, hp.img_size, 1)),
            Conv2D(96, kernel_size=(3, 3), activation='relu', padding='same'),
            MaxPool2D(pool_size=(2, 2)),
            Dropout(0.5),
            
            Conv2D(192, kernel_size=(3, 3), activation='relu', padding='same'),
            Conv2D(192, kernel_size=(3, 3), activation='relu', padding='same'),
            MaxPool2D(pool_size=(2, 2)),
            Dropout(0.5),
            
            Conv2D(384, kernel_size=(3, 3), activation='relu', padding='same'),
            Conv2D(384, kernel_size=(3, 3), activation='relu', padding='same'),
            MaxPool2D(pool_size=(2, 2)),
            Dropout(0.5),
            
            Flatten(),
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(256, activation='relu'),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ]

    def call(self, x):
        for layer in self.architecture:
            x = layer(x)
        return x

    @staticmethod
    def loss_fn(labels, predictions):
        return tf.keras.losses.binary_crossentropy(labels, predictions)