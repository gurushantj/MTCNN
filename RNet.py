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
                                        name="conv3",
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
        self.trainCLSAcc = tf.keras.metrics.CategoricalAccuracy()
        self.trainBBAcc = tf.keras.metrics.MeanSquaredError("mse")

        self.trainFileWriter = tf.summary.create_file_writer(logdir="visual/")

        self.clsLoss = tf.keras.metrics.Mean(name="clsLoss")
        self.gradients = tf.keras.metrics.Mean(name="gradients")
        self.bbLoss = tf.keras.metrics.Mean(name="bbLoss")


    def calculateCrossEntropyLoss(self, output, clsLabel):
        return self.crossEntropyLoss(output, clsLabel)

    def calculateMeanSqrdLoss(self, output, bbLabel):
        return self.meanSqrdLoss(output, bbLabel)

    @tf.function
    def epoch(self, input, clsLabel, bbLabel, losses, trainPart):
        batch = input.shape[0]
        input = tf.reshape(input, [batch, 24, 24, 3])
        input = tf.image.random_flip_left_right(input)
        input = tf.image.random_flip_up_down(input)
        mask = tf.not_equal(tf.reduce_sum(bbLabel, axis=-1), 0)

        with tf.GradientTape() as tape:
            output = self.model(input)
            output[1] = tf.boolean_mask(output[1], mask)
            bbLabel = tf.boolean_mask(bbLabel, mask)

            loss = tf.case([(tf.equal(trainPart, 0), lambda: self.crossEntropyLoss(clsLabel,output[0])),
                            (tf.equal(trainPart, 1), lambda: self.meanSqrdLoss(output[1], bbLabel))
                            ], exclusive=True)
            clsLoss = tf.case(
                [(tf.equal(trainPart, 0), lambda: loss + tf.cast(tf.add_n([losses[0], losses[1]]), dtype=tf.float32)),
                 (tf.equal(trainPart, 1),
                  lambda: 0.5 * (loss + tf.cast(tf.add_n([losses[0], losses[2]]), dtype=tf.float32)))
                 ], exclusive=True)

        gradient = tape.gradient(clsLoss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradient, self.model.trainable_variables))
        tf.case([(tf.equal(trainPart, 0), lambda: self.trainCLSAcc(clsLabel, output[0])),
                 (tf.equal(trainPart, 1), lambda: self.trainBBAcc(bbLabel, output[1]))],
                exclusive=True
                )

        tf.case([(tf.equal(trainPart, 0), lambda: self.clsLoss(loss)),
                 (tf.equal(trainPart, 1), lambda: self.bbLoss(loss))
                 ],
                exclusive=True
                )
        self.gradients.update_state(gradient)

    def _parseDataset(self, records):
        records = tf.io.parse_example(records, self.features)
        image = records["image"]
        label = records["label"]
        boundingBox = records["boundingBox"]
        label = tf.io.decode_raw(label, tf.float32)
        boundingBox = tf.io.decode_raw(boundingBox, tf.float32)
        image = tf.io.decode_raw(image, out_type=tf.float32)
        image = tf.cast(image, tf.float32)
        return image, label, boundingBox

    def readRecordDataSet(self, path):
        dataset = tf.data.Dataset.from_tensor_slices(path)
        dataset = dataset.interleave(lambda x: tf.data.TFRecordDataset(x),
                                     num_parallel_calls=tf.data.experimental.AUTOTUNE).cache()
        dataset = dataset.shuffle(buffer_size=6144).batch(6144).prefetch(4).map(self._parseDataset,
                                                                                num_parallel_calls=tf.data.experimental.AUTOTUNE).cache()
        return dataset

    def train(self, epoch, dataset_cls):
        for i in range(epoch):
            batch_count = 0
            epoch_execution_time = 0
            start = time.time()
            data_reading_start_time = time.time()

            self.trainCLSAcc.reset_states()
            self.clsLoss.reset_states()
            self.bbLoss.reset_states()
            self.trainBBAcc.reset_states()
            self.gradients.reset_states()

            for image, label, boundingBoxLabel in dataset_cls:
                randNum = np.random.randint(0, 2)
                batch_count += 1
                start_epoch_time = time.time()
                l2 = self.model.get_layer("conv4").losses[0]
                l3 = self.model.get_layer("conv5-1").losses[0]
                l4 = self.model.get_layer("conv5-2").losses[0]
                self.epoch(tf.constant(image), tf.constant(label), tf.constant(boundingBoxLabel),
                                      tf.stack([l2, l3, l4]),
                                      tf.constant(randNum))
                end_epoch_time = time.time()
                epoch_execution_time += (end_epoch_time - start_epoch_time)
            data_reading_end_time = time.time()
            end = time.time()

            print(
                "Epoch count :{0} epoch execution time: {1},cls loss {2},cls accuracy {3},BB loss {4},BB accuracy {5} , model exeution time {6},"
                "Data reading time {7}".format(i, (end - start), self.clsLoss.result(), self.trainCLSAcc.result(),
                                               self.bbLoss.result(), self.trainBBAcc.result(),
                                               epoch_execution_time, (data_reading_end_time - data_reading_start_time)))

            if i % 1000 == 0:
                with self.trainFileWriter.as_default():
                    tf.summary.scalar("BB Loss", self.bbLoss.result(), i)
                    tf.summary.scalar("CLS Loss", self.clsLoss.result(), i)
                    tf.summary.scalar("CLS Accuracy", self.trainCLSAcc.result(), i)
                    tf.summary.scalar("Train BB Accuracy", self.trainBBAcc.result(), i)
                    tf.summary.histogram("Gradients", self.gradients.result(), i)
                    layers = self.model.layers
                    names = ["weight", "bias"]
                    for layer in layers:
                        weights = layer.get_weights()
                        for name, weight in zip(names, weights):
                            tf.summary.histogram(name=layer.name + "_" + name, data=weight, step=i)

            if i % 1000 == 0:
                self.model.save(filepath="model/{0}/24".format(i))
                self.model.save_weights(filepath="model_wts/{0}/24".format(i))

        if i % 1000 == 0:
            self.model.save(filepath="model/{0}/12".format(i))
            self.model.save_weights(filepath="model_wts/{0}/12".format(i))