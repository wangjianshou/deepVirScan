import numpy as np
import tensorflow as tf
import tensorboard
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

from data_processing import generate_r1r2

class BatchNormGraph(layers.Layer):
  def __init__(self, axis=-1, epsilon=0.001, center=True, scale=True):
    super.__init__()
    self.epsilon = epsilon
    self.scale = scale
    self.center = center
    self.axis = axis
  def build(self, input_shape):
    ndims = len(input_shape)
    if isinstance(self.axis, int):
      self.axis = [self.axis,]
    for idx, x in enumerate(self.axis):
      if x < 0:
        self.axis[idx] = ndims + x
    self.mean = self.add_weight()

class mini_model(keras.Model):
  def __init__(self, channel=64, kernel=3):
    super().__init__()
    self.conv = layers.Conv1D(channel, kernel, strides=1, activation='relu', padding='valid')
    self.pool = layers.MaxPooling1D(kernel, strides=kernel, padding='valid')
    self.global_pool = layers.GlobalAveragePooling1D()
  @tf.function
  def call(self, inputs):
    r1 = self.pool(self.conv(inputs[0]))
    r2 = self.pool(self.conv(inputs[1]))
    block = layers.concatenate([r1,r2], axis=-2)
    return self.global_pool(block)

'''
# 测试mini_model
tmp = (tf.random.normal([256, 150, 2]), tf.random.normal([256, 150, 2]), np.random.random_integers(0,1, 256))
d = tf.data.Dataset.from_tensor_slices(tmp).map(lambda x,y,z: ((x, y),z)).repeat().batch(128)
m = mini_model()
m.compile(optimizer=tf.optimizers.Adam(0.001),
          loss = 'binary_crossentropy', metrics=['acc'])
m.fit(d, steps_per_epoch=10, epochs=10,validation_data=d, validation_steps=5)
'''

class VirScan(keras.Model):
  def __init__(self):
    super().__init__()
    self.embed = layers.Embedding(5, 2, input_length=150)
    self.mini_5 = mini_model(128, 5)
    self.mini_7 = mini_model(128, 7)
    self.mini_9 = mini_model(128, 9)
    self.mini_11 = mini_model(64, 11)
    self.mini_13 = mini_model(64, 13)
    self.mini_15 = mini_model(64, 15)
    self.mini_17 = mini_model(64, 17)
    self.mini_19 = mini_model(64, 19)
    self.dense_1 = layers.Dense(512, activation='relu')
    self.norm_1 = layers.BatchNormalization(axis=-1)
    self.drop_1 = layers.Dropout(0.1)
    self.output_1 = layers.Dense(1, activation='sigmoid')
  @tf.function
  def call(self, inputs, training=True):
    embed = [self.embed(inputs[0]), self.embed(inputs[1])]
    f5 = self.mini_5(embed)
    f7 = self.mini_7(embed)
    f9 = self.mini_9(embed)
    f11 = self.mini_11(embed)
    f13 = self.mini_13(embed)
    f15 = self.mini_15(embed)
    f17 = self.mini_17(embed)
    f19 = self.mini_19(embed)
    block_all = layers.concatenate([f5,f7,f9,f11, f13, f15, f17, f19], axis=-1)
    dense = self.dense_1(block_all)
    norm = self.norm_1(dense, training=training)
    drop = self.drop_1(norm, training=training)
    return self.output_1(drop)

# 测试
model = VirScan()
train = generate_r1r2('../example/example.train.data', 1024)
validation = generate_r1r2('../example/example.validation.data', 256)
model.compile(optimizer=tf.optimizers.Adam(0.001),
              loss = 'binary_crossentropy', metrics=['acc'])
model.fit(train, steps_per_epoch=10, epochs=10,validation_data=validation, validation_steps=5)


'''
if __name__=='__main__':
  train_file = "../data/r1r2_train.data"
  validation_file = "../data/r1r2_validation.data"

  callbacksList = [keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=50),
                   keras.callbacks.TensorBoard(log_dir='../model/train_logs',
                   histogram_freq=1, embeddings_freq=10),
                   keras.callbacks.ModelCheckpoint(filepath="../model/deepVirScan.h5",
                   monitor='val_loss', save_best_only=True),
                  ]

  model.compile(optimizer=tf.optimizers.Adam(0.001),
                loss = 'binary_crossentropy', metrics=['acc'])


  history = model.fit(train, steps_per_epoch=1000, epochs=1000,
                      validation_data=validation,
                      validation_steps=20,
                      callbacks=callbacksList)
  '''
