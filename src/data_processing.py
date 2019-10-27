import numpy as np
import tensorflow as tf

def base2num(file, label):
  label = '\t'+ str(int(label))+'\n'
  trans = str.maketrans('ATCGN','01234')
  numfile = file+'.num'
  with open(numfile, 'wt') as fw:
    with open(file,'rt') as fr:
      for base in fr:
        fw.write(base.strip().translate(trans)+label)
  return numfile

def shuffleSample(file):
  with open(file, 'rt') as f:
    sample = np.array(f.readlines())
  index = np.arange(sample.shape[0])
  np.random.shuffle(index)
  shuffle_file = file+'.shuffle'
  with open(shuffle_file, 'w') as f:
    f.writelines(sample[index])
  return shuffle_file

def generate_predict(file, buffer_size):
  dataset = tf.data.TextLineDataset(file)
  dataset = dataset.map(lambda x: tf.strings.split(x, '\t'))
  dataset = dataset.map(lambda x: ((tf.strings.to_number(tf.strings.bytes_split(x[0])), tf.strings.to_number(tf.strings.bytes_split(x[1]))), tf.strings.to_number(x[2])))
  dataset = dataset.shuffle(buffer_size)
  return dataset

def generate_r1r2(file, buffer_size):
  dataset = tf.data.TextLineDataset(file)
  dataset = dataset.map(lambda x: tf.strings.split(x, '\t'))
  dataset = dataset.map(lambda x: ((tf.strings.to_number(tf.strings.bytes_split(x[0])), tf.strings.to_number(tf.strings.bytes_split(x[1]))), tf.strings.to_number(x[2])))
  dataset = dataset.shuffle(buffer_size).repeat()
  return dataset.batch(256)
