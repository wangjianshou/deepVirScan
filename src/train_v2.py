import argparse
import numpy as np
import tensorflow as tf
import tensorboard
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

from .data_processing import generate_r1r2
from .deepVirScan import mini_model,VirScan


def train_keras(train_file, validation_file, epochs, steps, v_steps):
  model = VirScan()
  train = generate_r1r2(train_file, 1024)
  validation = generate_r1r2(validation_file, 256)
  callbacksList = [keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=50),
                   keras.callbacks.TensorBoard(log_dir='model/tensorboard',
                   histogram_freq=1),
                  ]
  model.compile(optimizer=tf.optimizers.Adam(0.001),
                loss = 'binary_crossentropy', metrics=['acc'])
  history = model.fit(train, steps_per_epoch=steps, epochs=epochs,
                      validation_data=validation,
                      validation_steps=v_steps,
                      callbacks=callbacksList)
  ckpt = tf.train.Checkpoint(model=model)
  ckpt.save("model/deepVirScan2-ckpt")
  return history

def train(train_file, validation_file, epochs, steps, v_steps, retrain=False):
  model = VirScan()
  optimizer = tf.optimizers.Adam()
  accuracy = tf.metrics.BinaryAccuracy()
  mean_loss = tf.metrics.Mean()
  writer = tf.summary.create_file_writer("model/tensorboard")
  train = iter(generate_r1r2(train_file, 1024))
  validation = iter(generate_r1r2(validation_file, 256))
  if retrain:
    ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
    latest = tf.train.latest_checkpoint("model/deepVirScan-ckpt/")
    ckpt.restore(latest)
  for epoch in range(epochs):
    for step in range(steps):
      with tf.GradientTape() as tape:
        x,y = next(train)
        y_pred = model(x, training=True)
        loss = tf.losses.binary_crossentropy(y, y_pred) + tf.reduce_sum(model.losses)
      grad = tape.gradient(loss, model.trainable_variables)
      optimizer.apply_gradients(grads_and_vars=zip(grad, model.trainable_variables))
      accuracy.update_state(y, y_pred)
      mean_loss.update_state(loss)
    with writer.as_default():
      tf.summary.scalar('train_loss-epoch', mean_loss.result(), step=epoch)
      tf.summary.scalar('train_accyracy-epoch', accuracy.result(), step=epoch)
    tf.print("{}/{}\t".format(epoch, epochs), "train_loss-epoch: ", mean_loss.result())
    tf.print("{}/{}\t".format(epoch, epochs), "train_accyracy-epoch:", accuracy.result())
    accuracy.reset_states()
    mean_loss.reset_states()
    for v_step in range(v_steps):
      vx, vy = next(validation)
      vy_pred = model(vx, training=False)
      mean_loss.update_state(tf.losses.binary_crossentropy(vy, vy_pred))
      accuracy.update_state(vy, vy_pred)
    with writer.as_default():
      tf.summary.scalar('validation_loss', mean_loss.result(), step=epoch)
      tf.summary.scalar('validation_accuracy', accuracy.result(), step=epoch)
    tf.print("{}/{}\t".format(epoch, epochs), "validation_loss-epoch: ", mean_loss.result())
    tf.print("{}/{}\t".format(epoch, epochs), "validation_accyracy: ", accuracy.result())
    accuracy.reset_states()
    mean_loss.reset_states()
  ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
  ckpt.save("model/deepVirScan-ckpt/deepVirScan")
  tf.saved_model.save(model, "model/deepVirScan/", signatures=model.call.get_concrete_function(
                      (tf.TensorSpec(shape=(None,150), dtype=tf.float32), 
                       tf.TensorSpec(shape=(None,150), dtype=tf.float32)), False)
                     )
def parseArgs():
  argparser = argparse.ArgumentParser(description="Train the model scanning reads")
  argparser.add_argument("--retrain", "-r", required=False, default=False, action="store_true")
  argparser.add_argument("--epochs", "-e", required=False, default=10, type=int)
  argparser.add_argument("--steps", "-s", required=False, default=5, type=int)
  argparser.add_argument("--vsteps", "-vs", required=False, default=5, type=int)
  return argparser.parse_args()

if __name__=='__main__':
  args = parseArgs()
  train_file = "data/r1r2_train.data"
  validation_file = "data/r1r2_validation.data"
  train(train_file, validation_file, args.epochs, args.steps, args.vsteps, args.retrain)




