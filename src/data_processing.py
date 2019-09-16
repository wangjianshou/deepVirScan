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

def generateData(file):
  dataset = tf.data.TextLineDataset(file)
  dataset = dataset.map(lambda i: tf.strings.to_number(tf.strings.bytes_split(i), 'int32'))
  dataset = dataset.batch(128).map(lambda i: (tf.one_hot(i[:,0:-1],4), i[:,-1]))
  return dataset.shuffle(256).repeat()


