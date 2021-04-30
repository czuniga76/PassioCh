

# Model description
# MobileNetV2 feature extractor with input size 224 x 224 and imagenet weights, and include_top = False.
# Outputs an L2-normed unit vector.

import tensorflow as tf
import os

os.mkdir("model")

def feature_extractor(inputs):
  # Use MobileNetV2 as feature extractor with imagenet weights
  feature_extractor = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3),
                                               include_top=False,
                                               weights='imagenet')(inputs)
  return feature_extractor

class DenseL2Norm(tf.keras.layers.Layer):
  # output layer is an L2 unit vector.
  def __init__(self,units):
    super().__init__()
    self.units = units
  
  def build(self,input_shape):
    w_init = tf.random_normal_initializer()
    self.w = tf.Variable(name="kernel",
                         initial_value = w_init(shape=(input_shape[-1],self.units),dtype="float32"),trainable=True)
    
  def call(self,inputs):
    y = tf.matmul(inputs,self.w)
    return y/tf.norm(y)
    
  def get_config(self):
    base_config = super().get_config()
    return {**base_config,"units":self.units}

def define_compile_model():
  inputs = tf.keras.layers.Input(shape=(224,224,3))
  
  mobilenetv2_features = feature_extractor(inputs) 
  x = tf.keras.layers.GlobalAveragePooling2D()(mobilenetv2_features)
  x = tf.keras.layers.Flatten()(x)
  outputUnitVector = DenseL2Norm(10)(x)

  
  model = tf.keras.Model(inputs=inputs, outputs = outputUnitVector)
 
  model.compile(optimizer='Adam', 
                loss='mse',
                metrics = ['accuracy'])
  
  return model

model = define_compile_model()
model.summary()

model.save("./model/denseL2.h5")