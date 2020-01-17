import tensorflow.keras as tfk
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


(x_train, y_train), (x_test, y_test) = tfk.datasets.fashion_mnist.load_data()
model = tfk.models.load_model('fashion_mnist.h5')


class_names = {0: 'T-shirt/top', 1: 'Trouser', 2: 'Pullover', 3: 'Dress', 4: 'Coat',
               5: 'Sandal', 6: 'Shirt', 7: 'Sneaker', 8: 'Bag', 9: 'Ankle boot'}

plt.figure(figsize=(10, 6))
test = np.random.randint(0, 9999, size=(8,))
print(test)
sub = 1
for i in test:
    image = np.expand_dims(x_test[i], axis=0)
    image = tf.dtypes.cast(image, tf.float16)
    image = np.reshape(image, [1, 28, 28, 1])
    prediction_vals = model.predict(image)
    pred = np.argpartition(prediction_vals, -2)[:, -2:]
    y_true = np.int(y_test[i])
    plt.subplot(2, 4, sub)
    plt.imshow(x_test[i], cmap='binary')
    one = class_names[np.int(pred[:, 1])]
    two = class_names[np.int(pred[:, 0])]
    plt.xlabel(f"Top 1: {one}\nTop 2: {two}\
    \nTrue: {class_names[y_true]}")
    plt.xticks([])
    plt.yticks([])
    sub += 1
plt.show()
