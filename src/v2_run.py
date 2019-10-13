import sys
import numpy as np
import tensorflow as tf
from data_processing import generate_predict
from deepVirScan import *

model = VirScan()
data = generate_predict(sys.argv[1])
ckpt = tf.train.Checkpoint(model=model)
latest = tf.train.latest_checkpoint("../model/v2/")
ckpt.restore(latest)
r = model.predict(data)
np.savetxt(r, "predict.result")
