import numpy as np
import tensorflow as tf
import tensorboard
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

train_file = "../data/r1r2_train.data"
validation_file = "../data/r1r2_validation.data"

#包含Embedding层
def generate(file, buffer_size):
  dataset = tf.data.TextLineDataset(file)
  dataset = dataset.map(lambda i: tf.strings.to_number(tf.strings.bytes_split(i), 'int32'))
  dataset = dataset.batch(128).map(lambda i: (i[:,0:-1], i[:,-1]))
  return dataset.shuffle(buffer_size).repeat()

def generate_r1r2(file, buffer_size):
  dataset = tf.data.TextLineDataset(file)
  dataset = dataset.map(lambda x: tf.strings.split(x, '\t'))
  dataset = dataset.map(lambda x: ((tf.strings.to_number(tf.strings.bytes_split(x[0])), tf.strings.to_number(tf.strings.bytes_split(x[1]))), tf.strings.to_number(x[2])))
  dataset = dataset.shuffle(buffer_size).repeat()
  return dataset.batch(128)

train = generate_r1r2(train_file, 1024)
validation = generate_r1r2(validation_file, 256)

r1 = keras.Input(shape=(150, ))
r2 = keras.Input(shape=(150,))
embed = layers.Embedding(5, 2, input_length=150)
r1_vec = embed(r1)
r2_vec = embed(r2)

# 3 bases
r1_conv_f3 = layers.Conv1D(128, 3, strides=1, activation='relu', padding='valid')(r1_vec)
r1_Pool_f3 = layers.MaxPooling1D(3, strides=3, padding='valid')(r1_conv_f3)

r2_conv_f3 = layers.Conv1D(128, 3, strides=1, activation='relu', padding='valid')(r2_vec)
r2_Pool_f3 = layers.MaxPooling1D(3, strides=3, padding='valid')(r2_conv_f3)

block_f3 = layers.concatenate([r1_Pool_f3, r2_Pool_f3], axis=-2)
f3_out = layers.GlobalAveragePooling1D()(block_f3)

# 4 bases

r1_conv_f4 = layers.Conv1D(128, 4, strides=1, activation='relu', padding='valid')(r1_vec)
r1_Pool_f4 = layers.MaxPooling1D(4, strides=4, padding='valid')(r1_conv_f4)

r2_conv_f4 = layers.Conv1D(128, 4, strides=1, activation='relu', padding='valid')(r2_vec)
r2_Pool_f4 = layers.MaxPooling1D(4, strides=4, padding='valid')(r2_conv_f4)

block_f4 = layers.concatenate([r1_Pool_f4, r2_Pool_f4], axis=-2)
f4_out = layers.GlobalAveragePooling1D()(block_f4)

# 5 bases
r1_conv_f5 = layers.Conv1D(64, 5, strides=1, activation='relu', padding='valid')(r1_vec)
r1_Pool_f5 = layers.MaxPooling1D(5, strides=5, padding='valid')(r1_conv_f5)

r2_conv_f5 = layers.Conv1D(64, 5, strides=1, activation='relu', padding='valid')(r2_vec)
r2_Pool_f5 = layers.MaxPooling1D(5, strides=5, padding='valid')(r2_conv_f5)

block_f5 = layers.concatenate([r1_Pool_f5, r2_Pool_f5], axis=-2)
f5_out = layers.GlobalAveragePooling1D()(block_f5)

# 6 bases
r1_conv_f6 = layers.Conv1D(64, 6, strides=1, activation='relu', padding='valid')(r1_vec)
r1_Pool_f6 = layers.MaxPooling1D(6, strides=6, padding='valid')(r1_conv_f6)

r2_conv_f6 = layers.Conv1D(64, 6, strides=1, activation='relu', padding='valid')(r2_vec)
r2_Pool_f6 = layers.MaxPooling1D(6, strides=6, padding='valid')(r2_conv_f6)

block_f6 = layers.concatenate([r1_Pool_f6, r2_Pool_f6], axis=-2)
f6_out = layers.GlobalAveragePooling1D()(block_f6)

# 7 bases
r1_conv_f7 = layers.Conv1D(64, 7, strides=1, activation='relu', padding='valid')(r1_vec)
r1_Pool_f7 = layers.MaxPooling1D(7, strides=7, padding='valid')(r1_conv_f7)

r2_conv_f7 = layers.Conv1D(64, 7, strides=1, activation='relu', padding='valid')(r2_vec)
r2_Pool_f7 = layers.MaxPooling1D(7, strides=7, padding='valid')(r2_conv_f7)

block_f7 = layers.concatenate([r1_Pool_f7, r2_Pool_f7], axis=-2)
f7_out = layers.GlobalAveragePooling1D()(block_f7)

# 8 bases
r1_conv_f8 = layers.Conv1D(64, 8, strides=1, activation='relu', padding='valid')(r1_vec)
r1_Pool_f8 = layers.MaxPooling1D(8, strides=8, padding='valid')(r1_conv_f8)

r2_conv_f8 = layers.Conv1D(64, 8, strides=1, activation='relu', padding='valid')(r2_vec)
r2_Pool_f8 = layers.MaxPooling1D(8, strides=8, padding='valid')(r2_conv_f8)

block_f8 = layers.concatenate([r1_Pool_f8, r2_Pool_f8], axis=-2)
f8_out = layers.GlobalAveragePooling1D()(block_f8)

# 9 bases
r1_conv_f9 = layers.Conv1D(64, 9, strides=1, activation='relu', padding='valid')(r1_vec)
r1_Pool_f9 = layers.MaxPooling1D(9, strides=9, padding='valid')(r1_conv_f9)

r2_conv_f9 = layers.Conv1D(64, 9, strides=1, activation='relu', padding='valid')(r2_vec)
r2_Pool_f9 = layers.MaxPooling1D(9, strides=9, padding='valid')(r2_conv_f9)

block_f9 = layers.concatenate([r1_Pool_f9, r2_Pool_f9], axis=-2)
f9_out = layers.GlobalAveragePooling1D()(block_f9)

# 10 bases
r1_conv_f10 = layers.Conv1D(64, 10, strides=1, activation='relu', padding='valid')(r1_vec)
r1_Pool_f10 = layers.MaxPooling1D(10, strides=10, padding='valid')(r1_conv_f10)

r2_conv_f10 = layers.Conv1D(64, 10, strides=1, activation='relu', padding='valid')(r2_vec)
r2_Pool_f10 = layers.MaxPooling1D(10, strides=10, padding='valid')(r2_conv_f10)

block_f10 = layers.concatenate([r1_Pool_f10, r2_Pool_f10], axis=-2)
f10_out = layers.GlobalAveragePooling1D()(block_f10)

# 11 bases
r1_conv_f11 = layers.Conv1D(64, 11, strides=1, activation='relu', padding='valid')(r1_vec)
r1_Pool_f11 = layers.MaxPooling1D(11, strides=11, padding='valid')(r1_conv_f11)

r2_conv_f11 = layers.Conv1D(64, 11, strides=1, activation='relu', padding='valid')(r2_vec)
r2_Pool_f11 = layers.MaxPooling1D(11, strides=11, padding='valid')(r2_conv_f11)

block_f11 = layers.concatenate([r1_Pool_f11, r2_Pool_f11], axis=-2)
f11_out = layers.GlobalAveragePooling1D()(block_f11)

# 12 bases
r1_conv_f12 = layers.Conv1D(64, 12, strides=1, activation='relu', padding='valid')(r1_vec)
r1_Pool_f12 = layers.MaxPooling1D(12, strides=12, padding='valid')(r1_conv_f12)

r2_conv_f12 = layers.Conv1D(64, 12, strides=1, activation='relu', padding='valid')(r2_vec)
r2_Pool_f12 = layers.MaxPooling1D(12, strides=12, padding='valid')(r2_conv_f12)

block_f12 = layers.concatenate([r1_Pool_f12, r2_Pool_f12], axis=-2)
f12_out = layers.GlobalAveragePooling1D()(block_f12)

# 13 bases
r1_conv_f13 = layers.Conv1D(64, 13, strides=1, activation='relu', padding='valid')(r1_vec)
r1_Pool_f13 = layers.MaxPooling1D(13, strides=13, padding='valid')(r1_conv_f13)

r2_conv_f13 = layers.Conv1D(64, 13, strides=1, activation='relu', padding='valid')(r2_vec)
r2_Pool_f13 = layers.MaxPooling1D(13, strides=13, padding='valid')(r2_conv_f13)

block_f13 = layers.concatenate([r1_Pool_f13, r2_Pool_f13], axis=-2)
f13_out = layers.GlobalAveragePooling1D()(block_f13)

# 14 bases
r1_conv_f14 = layers.Conv1D(64, 14, strides=1, activation='relu', padding='valid')(r1_vec)
r1_Pool_f14 = layers.MaxPooling1D(14, strides=14, padding='valid')(r1_conv_f14)

r2_conv_f14 = layers.Conv1D(64, 14, strides=1, activation='relu', padding='valid')(r2_vec)
r2_Pool_f14 = layers.MaxPooling1D(14, strides=14, padding='valid')(r2_conv_f14)

block_f14 = layers.concatenate([r1_Pool_f14, r2_Pool_f14], axis=-2)
f14_out = layers.GlobalAveragePooling1D()(block_f14)

# 15 base
r1_conv_f15 = layers.Conv1D(64, 15, strides=1, activation='relu', padding='valid')(r1_vec)
r1_Pool_f15 = layers.MaxPooling1D(15, strides=15, padding='valid')(r1_conv_f15)

r2_conv_f15 = layers.Conv1D(64, 15, strides=1, activation='relu', padding='valid')(r2_vec)
r2_Pool_f15 = layers.MaxPooling1D(15, strides=15, padding='valid')(r2_conv_f15)

block_f15 = layers.concatenate([r1_Pool_f15, r2_Pool_f15], axis=-2)
f15_out = layers.GlobalAveragePooling1D()(block_f15)

# 16 bases 
r1_conv_f16 = layers.Conv1D(64, 16, strides=1, activation='relu', padding='valid')(r1_vec)
r1_Pool_f16 = layers.MaxPooling1D(16, strides=16, padding='valid')(r1_conv_f16)

r2_conv_f16 = layers.Conv1D(64, 16, strides=1, activation='relu', padding='valid')(r2_vec)
r2_Pool_f16 = layers.MaxPooling1D(16, strides=16, padding='valid')(r2_conv_f16)

block_f16 = layers.concatenate([r1_Pool_f16, r2_Pool_f16], axis=-2)
f16_out = layers.GlobalAveragePooling1D()(block_f16)

# 17 bases
r1_conv_f17 = layers.Conv1D(64, 17, strides=1, activation='relu', padding='valid')(r1_vec)
r1_Pool_f17 = layers.MaxPooling1D(17, strides=17, padding='valid')(r1_conv_f17)

r2_conv_f17 = layers.Conv1D(64, 17, strides=1, activation='relu', padding='valid')(r2_vec)
r2_Pool_f17 = layers.MaxPooling1D(17, strides=17, padding='valid')(r2_conv_f17)

block_f17 = layers.concatenate([r1_Pool_f17, r2_Pool_f17], axis=-2)
f17_out = layers.GlobalAveragePooling1D()(block_f17)

# 18 bases
r1_conv_f18 = layers.Conv1D(64, 18, strides=1, activation='relu', padding='valid')(r1_vec)
r1_Pool_f18 = layers.MaxPooling1D(18, strides=18, padding='valid')(r1_conv_f18)

r2_conv_f18 = layers.Conv1D(64, 18, strides=1, activation='relu', padding='valid')(r2_vec)
r2_Pool_f18 = layers.MaxPooling1D(18, strides=18, padding='valid')(r2_conv_f18)

block_f18 = layers.concatenate([r1_Pool_f18, r2_Pool_f18], axis=-2)
f18_out = layers.GlobalAveragePooling1D()(block_f18)

# 20 bases
r1_conv_f20 = layers.Conv1D(64, 20, strides=1, activation='relu', padding='valid')(r1_vec)
r1_Pool_f20 = layers.MaxPooling1D(20, strides=20, padding='valid')(r1_conv_f20)

r2_conv_f20 = layers.Conv1D(64, 20, strides=1, activation='relu', padding='valid')(r2_vec)
r2_Pool_f20 = layers.MaxPooling1D(20, strides=20, padding='valid')(r2_conv_f20)

block_f20 = layers.concatenate([r1_Pool_f20, r2_Pool_f20], axis=-2)
f20_out = layers.GlobalAveragePooling1D()(block_f20)


block = layers.concatenate([f3_out, f4_out, f5_out, f6_out, f7_out, f8_out, f9_out, f10_out, f11_out, f12_out, f13_out, f14_out, f15_out, f16_out, f17_out, f18_out, f20_out], axis=-1)
conv_dense = layers.Dense(512, activation='relu')(block)
norm_cnn = layers.BatchNormalization(axis=-1)(conv_dense)
drop_1 = layers.Dropout(0.1)(norm_cnn)
output = layers.Dense(1, activation='sigmoid')(conv_dense)

model = keras.Model([r1,r2], output)

callbacksList = [keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=50),
                 keras.callbacks.TensorBoard(log_dir='../model/train_logs', histogram_freq=1, embeddings_freq=10),
                 keras.callbacks.ModelCheckpoint(filepath="../model/deepVirScan.h5", monitor='val_loss', save_best_only=True),
                ]

model.compile(optimizer=tf.optimizers.Adam(0.001),
              loss = 'binary_crossentropy', metrics=['acc'])

history = model.fit(train, steps_per_epoch=1000, epochs=1000,
                    validation_data=validation,
                    validation_steps=20,
                    callbacks=callbacksList)

def loss_acc_plot(history): 
  loss = history.history['loss']
  val_loss = history.history['val_loss']
  acc = history.history['acc']
  val_acc = history.history['val_acc']
  epochs = range(1, len(loss)+1)
  plt.figure(figsize=(10,20))
  plt.subplot(2,1,1)
  plt.plot(epochs, acc, 'bo', label='Training acc') 
  plt.plot(epochs, val_acc, 'b', label='Validation acc') 
  plt.title('Training and validation accuracy') 
  plt.legend()
  plt.subplot(2,1,2)
  plt.plot(epochs, loss, 'bo', label='Training loss')
  plt.plot(epochs, val_loss, 'b', label='Validation loss')
  plt.title("Training and Validation loss")
  plt.legend()
  plt.savefig('../model/loss_acc.png', dpi=300)

loss_acc_plot(history)
