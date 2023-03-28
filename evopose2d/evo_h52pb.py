import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
import tensorflow as tf

model_name = 'M'

# Convert weights from .h5 format to .pb
model_path = 'models/evopose2d_{}_f32.h5'.format(model_name)
model = tf.keras.models.load_model(model_path)
tf.saved_model.save(model, "models/evopose2d_{}_f32".format(model_name))
