# Custom L1 Distance Layer Module
# WHY DO WE NEED THIS : its needed to load the custom model

# Import dependencies
import tensorflow as tf
from tensorflow.keras.layers import Layer
# from tensorflow import keras
# from keras.layers import Layer

# Custom L1 Distance Layer from our previous notebook
class L1Dist(Layer):
    
    # init method - inheritance 
    def __init__(self , **kwargs):
        super().__init__()
    
    # Magic happens here - similarity calculation 
    def call(self , input_embedding , validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)
