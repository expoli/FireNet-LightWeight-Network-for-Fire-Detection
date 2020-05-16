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

# xian shi ce shi tu xiang
import matplotlib.pyplot as plt

plt.imshow(img_tensor[0])
plt.show()

# 用一个输入张量和一个输出张量列表将模型实例化
from tensorflow.keras import models

layer_outputs = [layer.output for layer in model.layers[:9]]  # 提取前 9 层的输出
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)

activations = activation_model.predict(img_tensor)

# 将第 4 个通道可视化
first_layer_activation = activations[0]
print(first_layer_activation.shape)
plt.matshow(first_layer_activation[0, :, :, 4], cmap='viridis')
plt.show()

# 　将第 7 个通道可视化
plt.matshow(first_layer_activation[0, :, :, 7], cmap='viridis')
plt.show()
