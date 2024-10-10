import tensorflow as tf
from tensorflow.keras.layers import DepthwiseConv2D
from tensorflow.keras.models import load_model


class CustomDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, groups=1, **kwargs):
        super().__init__(**kwargs)
        self.groups = groups

model = tf.keras.models.load_model('/Users/ponmorin/Documents/Python_MachineLearning/Real_Bondro_Objectdetection/model/bondro_mark3/keras_model.h5', custom_objects={'DepthwiseConv2D': CustomDepthwiseConv2D})
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open("/Users/ponmorin/Documents/Python_MachineLearning/Real_Bondro_Objectdetection/model/bondro_mark3/bondro_modelV3", "wb").write(tflite_model)