from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.models import load_model

model = load_model('my_model.h5')
model.summary()

img_path = 'fire-baidu-38.jpg'

from tensorflow.keras.preprocessing import image
from keras.preprocessing.image import img_to_array
import numpy as np

img = image.load_img(img_path, target_size=(64, 64))
img_tensor = img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255.
print(img_tensor.shape)

layer_names = []
for layer in model.layers[:8]:
    layer_names.append(layer.name)

# xian shi ce shi tu xiang
import matplotlib.pyplot as plt

plt.imshow(img_tensor[0])
plt.show()

# 用一个输入张量和一个输出张量列表将模型实例化
from tensorflow.keras import models

layer_outputs = [layer.output for layer in model.layers[:9]]  # 提取前 9 层的输出
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)

activations = activation_model.predict(img_tensor)
images_per_row = 8
for layer_name, layer_activation in zip(layer_names, activations):
    n_features = layer_activation.shape[-1]
    size = layer_activation.shape[1]
    n_cols = n_features // images_per_row
    display_grid = np.zeros((size * n_cols, images_per_row * size))
    for col in range(n_cols):
        for row in range(images_per_row):
            channel_image = layer_activation[0, :, :,
                            col * images_per_row + row]

            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col * size: (col + 1) * size,
            row * size: (row + 1) * size] = channel_image

    scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1],
                        scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')
    plt.show()
