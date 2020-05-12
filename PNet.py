import tensorflow as tf
from tensorflow import keras
import numpy as np

from Constants import WEIGHT_DECAY, PNET_LEARNING_RATE
from mtcnn_util.mtcnn_util import MTCNNUtil
import time

tf.get_logger().setLevel("ERROR")
tf.random.set_seed(1)
np.random.seed(1)


class PNet:
    def __init__(self, trainingDataPath=None,weightDecay=WEIGHT_DECAY):
        self.optimizer = keras.optimizers.Adam(lr=PNET_LEARNING_RATE)
        self.crossEntropyLoss = keras.losses.CategoricalCrossentropy()
        self.meanSqrdLoss = keras.losses.MeanSquaredError()
        self.trainingDataPath = trainingDataPath
        self.features = {"image": tf.io.FixedLenFeature([], tf.string),
                         "label": tf.io.FixedLenFeature([], tf.string),
                         "boundingBox": tf.io.FixedLenFeature([], tf.string)
                         }
        self.weight_decay =  weightDecay




    def buildModel(self):
        input = keras.layers.Input(shape=[None, None, 3])

        layer_1 = keras.layers.Conv2D(filters=10, kernel_size=(3, 3), padding="valid", name="conv1",
                                      kernel_initializer=MTCNNUtil.get_kernal_initlizer())(input)
        layer_tmp = layer_1
        layer_1 = keras.layers.PReLU(name="PReLU1",
                                     alpha_initializer=MTCNNUtil.get_kernal_initlizer(),
                                     shared_axes=[1, 2])(layer_1)
        layer_1 = keras.layers.MaxPool2D(pool_size=(2, 2), padding="same", strides=(2, 2))(layer_1)

        layer_2 = keras.layers.Conv2D(filters=16, kernel_size=(3, 3), padding="valid", name="conv2",
                                      kernel_initializer=MTCNNUtil.get_kernal_initlizer())(
            layer_1)
        layer_2 = keras.layers.PReLU(name="PReLU2",
                                     alpha_initializer=MTCNNUtil.get_kernal_initlizer(),
                                     shared_axes=[1, 2])(layer_2)

        layer_3 = keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding="valid", name="conv3",
                                      kernel_initializer=MTCNNUtil.get_kernal_initlizer(),
                                      kernel_regularizer=MTCNNUtil.get_regularizer())(
            layer_2)
        layer_3 = keras.layers.PReLU(name="PReLU3",
                                     alpha_initializer=MTCNNUtil.get_kernal_initlizer(),
                                     shared_axes=[1, 2])(layer_3)

        layer_4 = keras.layers.Conv2D(filters=2, kernel_size=(1, 1),  name="conv4-1",
                                      kernel_initializer=MTCNNUtil.get_kernal_initlizer(),
                                      activation=MTCNNUtil.multiDimensionalSoftmax(axis=3),
                                      kernel_regularizer=MTCNNUtil.get_regularizer()
                                      )(layer_3)


        layer_5 = keras.layers.Conv2D(filters=4, kernel_size=(1, 1), name="conv4-2",
                                      activation=tf.keras.activations.linear,
                                      kernel_initializer=MTCNNUtil.get_kernal_initlizer(),
                                      kernel_regularizer=MTCNNUtil.get_regularizer())(layer_3)

        model = keras.models.Model(inputs=[input], outputs=[layer_4, layer_5,layer_tmp])
        self.trainCLSAcc = tf.keras.metrics.CategoricalAccuracy()
        self.trainBBAcc = tf.keras.metrics.MeanSquaredError("mse")
        self.trainFileWriter = tf.summary.create_file_writer(logdir="visual/")
        self.clsLoss = tf.keras.metrics.Mean(name="clsLoss")
        self.bbLoss = tf.keras.metrics.Mean(name="bbLoss")
        self.model = model

    def calculateCrossEntropyLoss(self, output, clsLabel):
        return self.crossEntropyLoss(output, clsLabel)

    def calculateMeanSqrdLoss(self, output, bbLabel):
        return self.meanSqrdLoss(output, bbLabel)

    @tf.function
    def epoch(self, input, clsLabel, bbLabel, losses, trainPart):
        batch = input.shape[0]
        input = tf.reshape(input, [batch, 12, 12, 3])
        input = tf.image.random_flip_left_right(input)
        input = tf.image.random_flip_up_down(input)
        clsLabel = tf.reshape(clsLabel, [batch, 1, 1, 2])
        bbLabel = tf.reshape(bbLabel, [batch, 1, 1, 4])
        # mask = tf.not_equal(tf.reduce_sum(bbLabel, axis=-1), 0)
        # mask = tf.reshape(mask,(-1,1))

        with tf.GradientTape() as tape:
            output = self.model(input)
            # output[1] = tf.boolean_mask(output[1], mask)
            # bbLabel = tf.boolean_mask(bbLabel, mask)

            loss = tf.case([(tf.equal(trainPart, 0), lambda: self.crossEntropyLoss(clsLabel,output[0])),
                            (tf.equal(trainPart, 1), lambda: self.meanSqrdLoss(bbLabel,output[1]))
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
        return gradient

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
        dataset = dataset.shuffle(buffer_size=10240).batch(10240).prefetch(4).map(self._parseDataset,
                                                                                num_parallel_calls=tf.data.experimental.AUTOTUNE).cache()
        return dataset

    def train(self, epoch, dataset_cls, dataset_bb):
        for i in range(epoch):
            batch_count = 0
            epoch_execution_time = 0
            data_reading_time = 0
            start = time.time()
            data_reading_start_time = time.time()

            self.trainCLSAcc.reset_states()
            self.clsLoss.reset_states()
            self.bbLoss.reset_states()
            self.trainBBAcc.reset_states()

            cls_iter = iter(dataset_cls)
            bb_iter = iter(dataset_bb)
            is_cls_end = False
            is_bb_end = False
            indx = 0
            while True:
                randNum = np.random.randint(0, 2)
                indx += 1
                if randNum == 0 and is_cls_end == False:
                    try:
                        image, label, boundingBoxLabel = next(cls_iter)
                    except StopIteration as exp:
                        is_cls_end = True
                elif is_bb_end == False and randNum == 1:
                    try:
                        image, label, boundingBoxLabel = next(bb_iter)
                    except StopIteration as exp:
                        is_bb_end = True

                if is_cls_end == True and is_bb_end == True:
                    break

                if is_cls_end == True and randNum == 0:
                    continue

                if is_bb_end == True and randNum == 1:
                    continue

                batch_count += 1
                start_epoch_time = time.time()
                l2 = self.model.get_layer("conv3").losses[0]
                l3 = self.model.get_layer("conv4-1").losses[0]
                l4 = self.model.get_layer("conv4-2").losses[0]
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
                    layers = self.model.layers
                    names = ["weight", "bias"]
                    for layer in layers:
                        weights = layer.get_weights()
                        for name, weight in zip(names, weights):
                            tf.summary.histogram(name=layer.name + "_" + name, data=weight, step=i)

            if i % 1000 == 0:
                self.model.save(filepath="model/{0}/12".format(i))
                self.model.save_weights(filepath="model_wts/{0}/12".format(i))

        if i % 1000 == 0:
            self.model.save(filepath="model/{0}/12".format(i))
            self.model.save_weights(filepath="model_wts/{0}/12".format(i))