import sys
import numpy as np
import tensorflow as tf
from data_processing import generate_r1r2

model = tf.keras.models.load_model("model/v1/deepVirScan.h5")
data = generate_r1r2(sys.argv[1])
r = model.predict(data)
np.savetxt(r, "predict.result")
