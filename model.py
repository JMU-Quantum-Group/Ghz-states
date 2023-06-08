import tensorflow as tf
from tensorflow.keras import Model
class ST_Model(Model):
    def __init__(self):
        super(ST_Model, self).__init__()
    def stmodel(self):
        stmodel = tf.keras.models.Sequential(
            [tf.keras.layers.Flatten(input_shape=[2]),
             tf.keras.layers.Dense(70, activation='relu'),
             tf.keras.layers.Dropout(0.1),
             tf.keras.layers.Dense(60, activation='relu'),
             tf.keras.layers.Dropout(0.1),
             tf.keras.layers.Dense(20, activation='relu'),
             tf.keras.layers.Dropout(0.1),
             tf.keras.layers.Dense(2, activation='softmax')
             ])
        return stmodel
'''
sl=SL_Model()
sl.compile(optimizer=tf.optimizers.SGD(lr=0.1),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=[tf.keras.metrics.sparse_categorical_accuracy])
'''