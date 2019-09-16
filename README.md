# deepVirScan
This project provides a deep learning method to detect virus reads from metagenomic data.

# Denendencies
`python3.6`  
`pandas`  
`numpy`  
`tensorflow 2.0.0`  

# usage
This is version is only used to test whether the convolutional neural network is feasible for seeking virus reads in metagenomics. But if you want to use this model, then the usage is below:
If the current directory is deepVirScan, then the command line is  
`python src/run.py example/example.train.data`  
Annother method is running code as below in python:
```
import sys
import numpy as np
import tensorflow as tf
from src.data_processing import generate_r1r2
model = tf.keras.models.load_model("model/deepVirScan.h5")
data = generate_r1r2("example/example.train.data")
r = model.predict(data)
```

# Version
0.01

# Contributors
Wang Jianshou



