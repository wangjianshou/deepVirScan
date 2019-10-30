import argparse
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
    self.conv_1 = layers.Conv1D(channel, kernel, strides=1, activation='relu', padding='valid')
    self.pool_1 = layers.MaxPooling1D(kernel, strides=kernel, padding='valid')
    self.conv_2 = layers.Conv1D(channel, kernel, strides=1, activation='relu', padding='valid')
    self.pool_2 = layers.MaxPooling1D(kernel, strides=kernel, padding='valid')
    self.global_pool = layers.GlobalAveragePooling1D()
  #@tf.function
  def call(self, inputs):
    r1 = self.pool_1(self.conv_1(inputs[0]))
    r2 = self.pool_2(self.conv_2(inputs[1]))
    block = layers.concatenate([r1,r2], axis=-2)
    return self.global_pool(block)


class VirScan(keras.Model):
  def __init__(self):
    super().__init__()
    self.embed = layers.Embedding(5, 2, input_length=150)
    self.mini_3 = mini_model(128, 3)
    self.mini_4 = mini_model(128, 4)
    self.mini_5 = mini_model(64, 5)
    self.mini_6 = mini_model(64, 6)
    self.mini_7 = mini_model(64, 7)
    self.mini_8 = mini_model(64, 8)
    self.mini_9 = mini_model(64, 9)
    self.mini_10 = mini_model(64, 10)
    self.mini_11 = mini_model(64, 11)
    self.mini_12 = mini_model(64, 12)
    self.mini_13 = mini_model(64, 13)
    self.mini_14 = mini_model(64, 14)
    self.mini_15 = mini_model(64, 15)
    self.mini_16 = mini_model(64, 16)
    self.mini_17 = mini_model(64, 17)
    self.mini_18 = mini_model(64, 18)
    self.mini_19 = mini_model(64, 19)
    self.mini_20 = mini_model(64, 20)
    self.mini_21 = mini_model(64, 21)
    self.mini_22 = mini_model(64, 22)
    self.mini_23 = mini_model(64, 23)
    self.mini_24 = mini_model(64, 24)
    self.mini_25 = mini_model(64, 25)
    self.dense_1 = layers.Dense(512, activation='relu')
    self.norm_1 = layers.BatchNormalization(axis=-1)
    self.drop_1 = layers.Dropout(0.1)
    self.dense_2 = layers.Dense(64, activation='relu')
    self.norm_2 = layers.BatchNormalization(axis=-1)
    self.drop_2 = layers.Dropout(0.1)
    self.output_1 = layers.Dense(1, activation='sigmoid')
  #@tf.function
  def call(self, inputs, training=None):
    embed = [self.embed(inputs[0]), self.embed(inputs[1])]
    f3 = self.mini_3(embed)
    f4 = self.mini_4(embed)
    f5 = self.mini_5(embed)
    f6 = self.mini_6(embed)
    f7 = self.mini_7(embed)
    f8 = self.mini_8(embed)
    f9 = self.mini_9(embed)
    f10 = self.mini_10(embed)
    f11 = self.mini_11(embed)
    f12 = self.mini_12(embed)
    f13 = self.mini_13(embed)
    f14 = self.mini_14(embed)
    f15 = self.mini_15(embed)
    f16 = self.mini_16(embed)
    f17 = self.mini_17(embed)
    f18 = self.mini_18(embed)
    f19 = self.mini_19(embed)
    f20 = self.mini_20(embed)
    '''
    f21 = self.mini_21(embed)
    f22 = self.mini_22(embed)
    f23 = self.mini_23(embed)
    f24 = self.mini_24(embed)
    f25 = self.mini_25(embed)
    '''
    block_all = layers.concatenate([f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,f16,f17,f18,f19,f20], axis=-1)
    dense_1 = self.dense_1(block_all)
    norm_1 = self.norm_1(dense_1, training=training)
    drop_1 = self.drop_1(norm_1, training=training)
    dense_2 = self.dense_2(block_all)
    norm_2 = self.norm_2(dense_2, training=training)
    drop_2 = self.drop_2(norm_2, training=training)
    return self.output_1(drop_2)






