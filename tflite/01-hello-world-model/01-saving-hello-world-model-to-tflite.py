import tensorflow as tf
import numpy as np
from tensorflow import keras

model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')
xs = np.array([-1.0,  0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

model.fit(xs, ys, epochs=500)

print(model.predict([10.0]))

import pathlib
#Export saved model
export_dir = '/Users/sairambs/Projects/ml/intro-to-tensorflow/tflite/01-hello-world-model/model'
tf.saved_model.save(model, export_dir)

#Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model(export_dir)
tflite_model = converter.convert()

#Save the model
tflite_model_file = pathlib.Path('/Users/sairambs/Projects/ml/intro-to-tensorflow/tflite/01-hello-world-model/model/hello-world-model.tflite')
tflite_model_file.write_bytes(tflite_model)