import numpy as np
import tensorflow as tf

from Constants import WEIGHT_DECAY
from enum import Enum
import os


class MTCNNUtil:

    STRIDE = 2
    WINDOW_SIZE = 12


    @staticmethod
    def genrate_bb(pnet_output,threshold,scale):
        bb_output = pnet_output[1]
        bb_output = bb_output[0]
        cls_output = pnet_output[0]
        cls_output = cls_output[0]
        cls_output = cls_output[:,:,1]

        bb_output = np.transpose(bb_output, axes=(1, 0, 2))
        cls_output = np.transpose(cls_output)
        # cls_output = imap
        # bb_output = reg
        (x, y) = np.where(cls_output >= threshold)
        bb = bb_output[(x, y)]
        score = cls_output[(x, y)]
        nx1 = x * MTCNNUtil.STRIDE
        ny1 = y * MTCNNUtil.STRIDE
        nx2 = nx1 + MTCNNUtil.WINDOW_SIZE
        ny2 = ny1 + MTCNNUtil.WINDOW_SIZE
        nx1 = nx1.astype(np.float)
        ny1 = ny1.astype(np.float)
        nx2 = nx2.astype(np.float)
        ny2 = ny2.astype(np.float)

        nx1 = np.expand_dims(np.fix((nx1 + 1) / scale), 1)
        ny1 = np.expand_dims(np.fix((ny1 + 1) / scale), 1)
        nx2 = np.expand_dims(np.fix(nx2 / scale), 1)
        ny2 = np.expand_dims(np.fix(ny2 / scale), 1)
        score = np.expand_dims(score, 1)
        boxes = np.hstack([nx1, ny1, nx2, ny2, score, bb])
        return boxes


    @staticmethod
    def bbreg(boundingbox, reg):
        if reg.shape[1] == 1:
            reg = np.reshape(reg, (reg.shape[2], reg.shape[3]))
        w = boundingbox[:, 2] - boundingbox[:, 0] + 1
        h = boundingbox[:, 3] - boundingbox[:, 1] + 1
        b1 = boundingbox[:, 0] + reg[:, 0] * w
        b2 = boundingbox[:, 1] + reg[:, 1] * h
        b3 = boundingbox[:, 2] + reg[:, 2] * w
        b4 = boundingbox[:, 3] + reg[:, 3] * h
        boundingbox[:, 0:4] = np.transpose(np.vstack([b1, b2, b3, b4]))
        return boundingbox

    @staticmethod
    def nms(boxes,threshold,method="UNION"):
        score_indexes = np.argsort(boxes[:,4])
        x1 = boxes[:,0]
        y1 = boxes[:,1]
        x2 = boxes[:,2]
        y2 = boxes[:,3]

        area = (x2-x1)*(y2-y1)
        pickup_ = np.zeros(boxes.shape[0],dtype=np.int)
        index = 0
        while score_indexes.size > 0 :
            indx = score_indexes[-1]
            pickup_[index] = indx
            box = boxes[indx]
            tmp_sorted_indexes = score_indexes[0:-1]
            x1_ = np.maximum(box[0],boxes[tmp_sorted_indexes,0])
            y1_ = np.maximum(box[1],boxes[tmp_sorted_indexes,1])
            x2_ = np.minimum(box[2],boxes[tmp_sorted_indexes,2])
            y2_ = np.minimum(box[3],boxes[tmp_sorted_indexes,3])
            area_ = np.maximum(0,(x2_ - x1_))*np.maximum(0,(y2_ - y1_))
            # area_ = (x2_ - x1_)*(y2_ - y1_)
            if method == "MIN":
                iou = area_ / np.minimum(area[indx] ,area[tmp_sorted_indexes])
            else:
                iou = area_/(area[indx]+area[tmp_sorted_indexes]-area_)
            # print(iou[np.where(iou > threshold)])
            score_indexes = score_indexes[np.where(iou <= threshold)]
            # print(index)
            index+=1
        return pickup_[0:index]

    @staticmethod
    def pad(total_boxes, w, h):
        tmpw = (total_boxes[:, 2] - total_boxes[:, 0] + 1).astype(np.int32)
        tmph = (total_boxes[:, 3] - total_boxes[:, 1] + 1).astype(np.int32)
        numbox = total_boxes.shape[0]

        dx = np.ones((numbox), dtype=np.int32)
        dy = np.ones((numbox), dtype=np.int32)
        edx = tmpw.copy().astype(np.int32)
        edy = tmph.copy().astype(np.int32)

        x = total_boxes[:, 0].copy().astype(np.int32)
        y = total_boxes[:, 1].copy().astype(np.int32)
        ex = total_boxes[:, 2].copy().astype(np.int32)
        ey = total_boxes[:, 3].copy().astype(np.int32)

        tmp = np.where(ex > w)
        edx.flat[tmp] = np.expand_dims(-ex[tmp] + w + tmpw[tmp], 1)
        ex[tmp] = w

        tmp = np.where(ey > h)
        edy.flat[tmp] = np.expand_dims(-ey[tmp] + h + tmph[tmp], 1)
        ey[tmp] = h

        tmp = np.where(x < 1)
        dx.flat[tmp] = np.expand_dims(2 - x[tmp], 1)
        x[tmp] = 1

        tmp = np.where(y < 1)
        dy.flat[tmp] = np.expand_dims(2 - y[tmp], 1)
        y[tmp] = 1
        return dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph

    @staticmethod
    def loadWeights(path):
        weightData = np.load(path, encoding='latin1', allow_pickle=True).item()
        return weightData

    @staticmethod
    def multiDimensionalSoftmax(axis):
        def loss(input):
            max_axis = tf.reduce_max(input, axis=axis, keepdims=True)
            input = input - max_axis
            input = tf.exp(input)
            denom = tf.reduce_sum(input, keepdims=True, axis=axis)
            return tf.math.divide(input, denom)
        return loss

    @staticmethod
    def get_weight_decay():
        return WEIGHT_DECAY

    @staticmethod
    def get_kernal_initlizer():
        return tf.keras.initializers.TruncatedNormal(stddev=MTCNNUtil.get_weight_decay())

    @staticmethod
    def get_regularizer():
        return tf.keras.regularizers.l2(l=MTCNNUtil.get_weight_decay())

    @staticmethod
    def setWeights(weightData,model,use_dict=False):
        for key, value in weightData.items():
            try:
                layer = model.get_layer(key)
            except ValueError as err:
                continue
            weights = []
            if type(value) == dict:
                for k,v in value.items():
                    weights.append(np.array(v))
            else:
                for v in value:
                    weights.append(np.array(v))
            try:
                layer_weights = layer.get_weights()
                for i in range(len(layer_weights)):
                    layer_weight = layer_weights[i]
                    weight = weights[i]
                    one_weights = np.ones(shape=(layer_weight.shape))
                    weight = np.multiply(one_weights, weight)
                    weights[i] = weight
                layer.set_weights(weights)
            except Exception as exp:
                raise exp



    @staticmethod
    def get_files(path):
        file_list = []
        for _,_,files in os.walk(path):
            for i in range(len(files)):
                file_list.append(os.path.join(path,files[i]))

        return file_list




MTCNNUtil.get_files("/Users/gurushant/PycharmProjects/MTCNN/weights")

class Mode(Enum):
    TRAINING = 0
    TESTING = 1

    @staticmethod
    def getMode(mode):
        return {
                Mode.TRAINING:"TRAINING",
                Mode.TESTING: "TESTING"
        }[mode]
