import tensorflow as tf
import pathlib

#Load Model
model = tf.keras.applications.MobileNetV2(weights="imagenet", input_shape=(224, 224, 3))

#Get the concrete function from keras model
run_model = tf.function(lambda x: model(x))

#Save the concrete function
concrete_func = run_model.get_concrete_function(tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))

#Save the model
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
tflite_model = converter.convert()

tflite_model_file = pathlib.Path('/Users/sairambs/Projects/ml/intro-to-tensorflow/tflite/03-concrete-functions/model/concrete-functions.tflite')
tflite_model_file.write_bytes(tflite_model)