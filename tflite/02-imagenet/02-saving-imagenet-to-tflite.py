import tensorflow as tf
import pathlib

model = tf.keras.applications.MobileNetV2(weights="imagenet", input_shape=(224, 224, 3))

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

tflite_model_file = pathlib.Path('/Users/sairambs/Projects/ml/intro-to-tensorflow/tflite/02-imagenet/model/imagenet.tflite')
tflite_model_file.write_bytes(tflite_model)