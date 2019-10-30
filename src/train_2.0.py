import argparse
import numpy as np
import tensorflow as tf
import tensorboard
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

from data_processing import generate_r1r2
from deepVirScan import mini_model,VirScan

# 测试mini_model
def test_mini_model():
  tmp = (tf.random.normal([256, 150, 2]), tf.random.normal([256, 150, 2]), np.random.random_integers(0,1, 256))
  d = tf.data.Dataset.from_tensor_slices(tmp).map(lambda x,y,z: ((x, y),z)).repeat().batch(128)
  m = mini_model()
  m.compile(optimizer=tf.optimizers.Adam(0.001),
          loss = 'binary_crossentropy', metrics=['acc'])
  m.fit(d, steps_per_epoch=10, epochs=10,validation_data=d, validation_steps=5)


# 测试
def test():
  model = VirScan()
  train = generate_r1r2('../example/example.train.data', 1024)
  validation = generate_r1r2('../example/example.validation.data', 256)
  model.compile(optimizer=tf.optimizers.Adam(0.001),
                loss = 'binary_crossentropy', metrics=['acc'])
  model.fit(train, steps_per_epoch=10, epochs=10,validation_data=validation, validation_steps=5)
  ckpt = tf.train.Checkpoint(model=model)
  ckpt.save("../model/v2/deepVirScan2")

def train_1():
  train_file = "../data/r1r2_train.data"
  validation_file = "../data/r1r2_validation.data"
  model = VirScan()
  train = generate_r1r2(train_file, 1024)
  validation = generate_r1r2(validation_file, 256)
  callbacksList = [keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=50),
                   keras.callbacks.TensorBoard(log_dir='../model/v2_train_logs',
                   histogram_freq=1),
                  ]
  model.compile(optimizer=tf.optimizers.Adam(0.001),
                loss = 'binary_crossentropy', metrics=['acc'])
  history = model.fit(train, steps_per_epoch=1000, epochs=1000,
                      validation_data=validation,
                      validation_steps=20,
                      callbacks=callbacksList)
  ckpt = tf.train.Checkpoint(model=model)
  ckpt.save("../model/v2/deepVirScan2")
  return history

def train():
  train_file = "../data/r1r2_train.data"
  validation_file = "../data/r1r2_validation.data"
  model = VirScan()
  train = generate_r1r2(train_file, 1024)
  validation = generate_r1r2(validation_file, 256)
  for epoch in range(1000):
    for step in range(1000):
      
    
  ckpt = tf.train.Checkpoint(model=model)
  ckpt.save("../model/v2/deepVirScan2")
  return history

def parseArgs():
  argparser = argparse.ArgumentParser(description="Train the model scanning reads")
  argparser.add_argument("--test", "-t", required=False, default=False, action="store_true")
  return argparser.parse_args()

if __name__=='__main__':
  args = parseArgs()
  if args.test:
    test()
  else:
    train()




