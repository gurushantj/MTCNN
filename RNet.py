import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
import time

from mtcnn_util.mtcnn_util import MTCNNUtil

tf.get_logger().setLevel("ERROR")
# tf.random.set_seed(1)
# np.random.seed(1)

class RNet:
    def __init__(self,weight_decay=4e-3,trainingDataPath=None):
        self.weight_decay = weight_decay
        self.optimizer = tf.keras.optimizers.Adam(lr=0.0001)
        self.crossEntropyLoss = tf.keras.losses.CategoricalCrossentropy()
        self.meanSqrdLoss = tf.keras.losses.MeanSquaredError()
        self.trainingDataPath = trainingDataPath
        self.features = {"image": tf.io.FixedLenFeature([], tf.string),
                         "label": tf.io.FixedLenFeature([], tf.string),
                         "boundingBox": tf.io.FixedLenFeature([], tf.string)
                         }

    def build_model(self):
        input = tf.keras.Input(shape=(24,24,3),dtype=tf.float16)
        layer1 = tf.keras.layers.Conv2D(filters=28,kernel_size=(3,3),name="conv1",kernel_initializer=MTCNNUtil.get_kernal_initlizer())(input)
        layer1 = tf.keras.layers.PReLU(name="prelu1",alpha_initializer=MTCNNUtil.get_kernal_initlizer())(layer1)
        layer1 = tf.keras.layers.MaxPool2D(pool_size=(3,3),strides=(2,2),name="pool1")(layer1)

        layer2 = tf.keras.layers.Conv2D(filters=48, kernel_size=(3, 3),  name="conv2",kernel_initializer=MTCNNUtil.get_kernal_initlizer())(layer1)
        layer2 = tf.keras.layers.PReLU(name="prelu2",alpha_initializer=MTCNNUtil.get_kernal_initlizer())(layer2)
        layer2 = tf.keras.layers.MaxPool2D(pool_size=(3,3),strides=(2,2),name="pool2",padding="SAME")(layer2)

        layer3 = tf.keras.layers.Conv2D(filters=64, kernel_size=(2, 2),
                                        name="conv3",kernel_regularizer= MTCNNUtil.get_regularizer(),
                                        kernel_initializer=MTCNNUtil.get_kernal_initlizer()
                                        )(layer2)
        layer3 = tf.keras.layers.PReLU(name="prelu3",alpha_initializer=MTCNNUtil.get_kernal_initlizer())(layer3)
        layer3 = tf.keras.layers.Flatten()(layer3)

        layer4 = tf.keras.layers.Dense(units=128,
                                       kernel_regularizer = MTCNNUtil.get_regularizer(),
                                       kernel_initializer=MTCNNUtil.get_kernal_initlizer(),
                                       name="conv4"
                                       )(layer3)
        layer4 = tf.keras.layers.PReLU(name="prelu4",alpha_initializer=MTCNNUtil.get_kernal_initlizer())(layer4)

        output1 = tf.keras.layers.Dense(units=2,activation=MTCNNUtil.multiDimensionalSoftmax(axis=1),
                                        kernel_regularizer=MTCNNUtil.get_regularizer(),
                                        kernel_initializer=MTCNNUtil.get_kernal_initlizer(),
                                        name="conv5-1"
                                        )(layer4)
        output2 = tf.keras.layers.Dense(units=4,
                                        kernel_regularizer=MTCNNUtil.get_regularizer(),
                                        kernel_initializer=MTCNNUtil.get_kernal_initlizer(),
                                        name = "conv5-2"
                                        )(layer4)

        model = tf.keras.Model(inputs=[input],outputs = [output1,output2])

        self.model = model
        self.trainCLSAcc = tf.keras.metrics.BinaryAccuracy("trainAccuracy",threshold=0.7)
        self.trainBBAcc = tf.keras.metrics.MeanSquaredError("mse")

        self.trainFileWriter = tf.summary.create_file_writer(logdir="visual/")

        self.clsLoss = tf.keras.metrics.Mean(name="clsLoss")
        self.bbLoss = tf.keras.metrics.Mean(name="bbLoss")