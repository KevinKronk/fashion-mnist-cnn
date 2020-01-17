import tensorflow.keras as tfk
import tensorflow.keras.layers as layers
import numpy as np


(x_train, y_train), (x_test, y_test) = tfk.datasets.fashion_mnist.load_data()
x_train = np.reshape(x_train, [60000, 28, 28, 1])
x_test = np.reshape(x_test, [10000, 28, 28, 1])

x_train = x_train / 255.0
x_test = x_test / 255.0

model = tfk.Sequential()
model.add(layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation=tfk.activations.relu,
                        input_shape=(28, 28, 1), padding='same'))
model.add(layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation=tfk.activations.relu,
                        padding='same'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Flatten())
model.add(layers.Dense(128, activation=tfk.activations.relu))
model.add(layers.Dense(10, activation=tfk.activations.softmax))

model.compile(optimizer=tfk.optimizers.Adam(),
              loss=tfk.losses.sparse_categorical_crossentropy,
              metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=128, epochs=10, verbose=1)
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)

print(loss, accuracy)

model.save('fashion_mnist.h5')
print('Model Saved.')
