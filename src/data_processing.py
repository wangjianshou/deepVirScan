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

