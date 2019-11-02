# deepVirScan
This project provides a deep learning method to detect virus reads from metagenomic data.

# Denendencies
`python3.6`  
`pandas`  
`numpy`  
`tensorflow 2.0.0`  

# Usage
This version is only used to test whether the convolutional neural network is feasible for seeking virus reads in metagenomics. But if you want to use this model to predict virus reads, then the usage is below.  
```
#If the current directory is deepVirScan, then the command line is 
python -m model.run_v2 example/example.train.data
```
If you want to get the source model to keep training with your own data, here's the code:
```
import tensorflow as tf
from src.data_processing import generate_predict
from src.deepVirScan import mini_model,VirScan

model = VirScan()
optimizer = tf.optimizers.Adam()
ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
latest = tf.train.latest_checkpoint("model/deepVirScan-ckpt/")
ckpt.restore(latest)
```

# Version
2.0



