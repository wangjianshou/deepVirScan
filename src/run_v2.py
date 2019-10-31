import sys
import os
import numpy as np
import tensorflow as tf
from .data_processing import generate_predict
from .deepVirScan import mini_model,VirScan 
'''
model = VirScan()
data = generate_predict(sys.argv[1], 1024)
ckpt = tf.train.Checkpoint(model=model)
latest = tf.train.latest_checkpoint("model/deepVirScan-ckpt/")
ckpt.restore(latest)
'''
data = iter(generate_predict(sys.argv[1], 1024))
model = tf.saved_model.load("model/deepVirScan")
result = open('result.txt', 'w')
for i in data:
  r = model.call(i, False)
  np.savetxt("predict.tmp", r)
  with open("predict.tmp") as f:
    result.write(f.read())
result.close()
os.system('rm predict.tmp')
