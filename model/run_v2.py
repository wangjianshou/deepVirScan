import sys
import os
import numpy as np
import tensorflow as tf
from src.data_processing import generate_predict
from src.deepVirScan import mini_model,VirScan 

'''
model = VirScan()
optimizer = tf.optimizers.Adam()
ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
latest = tf.train.latest_checkpoint("model/deepVirScan-ckpt/")
ckpt.restore(latest)
data = generate_predict(sys.argv[1], 1024)
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
