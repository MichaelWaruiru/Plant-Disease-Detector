import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf

tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.keras.backend.set_learning_phase(0)

from ml_model import PlantDiseaseModel
print("LOADING MODEL ONCE")
model = PlantDiseaseModel()