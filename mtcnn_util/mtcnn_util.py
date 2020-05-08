import numpy as np
import tensorflow as tf

from Constants import WEIGHT_DECAY
from enum import Enum
import os
import cv2


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

    # @staticmethod
    # def iou(box,boxes):
    #     x1 = boxes[:,0]
    #     y1 = boxes[:,1]
    #     x2 = boxes[:,2]
    #     y2 = boxes[:,3]
        x1 = boxes[0]
        y1 = boxes[1]
        x2 = boxes[2]
        y2 = boxes[3]
        #
        # width = x2-x1
        # height = y2-y1
        # width = np.maximum(width,1)
        # height = np.maximum(height,1)
        # area = width*height
        # box_area = np.maximum((box[2]-box[0]),1)*np.maximum((box[3]-box[1]),1)
        # nx1 = np.maximum(box[0],x1)
        # ny1 = np.maximum(box[1],y1)
        # nx2 = np.minimum(box[2], x2)
        # ny2 = np.minimum(box[3], y2)
        #
        # min_area = np.minimum(area,box_area)
        #
        # inter_area = (nx2-nx1)*(ny2-ny1)
        inter_area[inter_area < 0] = 0
        # iou = inter_area/(box_area+min_area-inter_area)
        # return iou
    #
    @staticmethod
    def iou(box, boxes):
        box_area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
        area = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
        xx1 = np.maximum(box[0], boxes[:, 0])
        yy1 = np.maximum(box[1], boxes[:, 1])
        xx2 = np.minimum(box[2], boxes[:, 2])
        yy2 = np.minimum(box[3], boxes[:, 3])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        inter = w * h
        ovr = inter / (box_area + area - inter)
        # ovr = inter / (2 * area - inter)
        return ovr

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

    @staticmethod
    def detect_face_12net(img, minsize, pnet, threshold, factor):

        factor_count = 0
        total_boxes = np.empty((0, 9))
        h = img.shape[0]
        w = img.shape[1]
        minl = np.amin([h, w])
        m = 12.0 / minsize
        minl = minl * m
        # creat scale pyramid
        scales = []
        while minl >= 12:
            scales += [m * np.power(factor, factor_count)]
            minl = minl * factor
            factor_count += 1

        # first stage
        for j in range(len(scales)):
            scale = scales[j]
            hs = int(np.ceil(h * scale))
            ws = int(np.ceil(w * scale))
            im_data = cv2.resize(img, (hs, ws))
            im_data = (im_data - 127.5) * (1. / 128.0)
            img_x = np.expand_dims(im_data, 0)
            out = pnet(img_x)
            boxes = MTCNNUtil.genrate_bb(out,
                                           scale,
                                           threshold)

            # inter-scale nms
            pick = MTCNNUtil.nms(boxes.copy(), 0.5, 'Union')
            if boxes.size > 0 and pick.size > 0:
                boxes = boxes[pick, :]
                total_boxes = np.append(total_boxes, boxes, axis=0)

        numbox = total_boxes.shape[0]
        if numbox > 0:
            pick = MTCNNUtil.nms(total_boxes.copy(), 0.7, 'Union')
            total_boxes = total_boxes[pick, :]
            regw = total_boxes[:, 2] - total_boxes[:, 0]
            regh = total_boxes[:, 3] - total_boxes[:, 1]
            qq1 = total_boxes[:, 0] + total_boxes[:, 5] * regw
            qq2 = total_boxes[:, 1] + total_boxes[:, 6] * regh
            qq3 = total_boxes[:, 2] + total_boxes[:, 7] * regw
            qq4 = total_boxes[:, 3] + total_boxes[:, 8] * regh
            total_boxes = np.transpose(np.vstack([qq1, qq2, qq3, qq4,
                                                  total_boxes[:, 4]]))
            total_boxes[:, 0:4] = np.fix(total_boxes[:, 0:4]).astype(np.int32)
        return total_boxes

    @staticmethod
    def detect_face_24net(img, minsize, pnet, rnet, threshold, factor):

        factor_count = 0
        total_boxes = np.empty((0, 9))
        h = img.shape[0]
        w = img.shape[1]
        minl = np.amin([h, w])
        m = 12.0 / minsize
        minl = minl * m
        # creat scale pyramid
        scales = []
        while minl >= 12:
            scales += [m * np.power(factor, factor_count)]
            minl = minl * factor
            factor_count += 1

        # first stage
        for j in range(len(scales)):
            scale = scales[j]
            hs = int(np.ceil(h * scale))
            ws = int(np.ceil(w * scale))
            im_data = cv2.resize(img, (ws, hs))
            im_data = (im_data - 127.5) * 0.0078125
            img_x = np.expand_dims(im_data, 0)
            out = pnet(img_x)
            out0 = out[0]
            out1 = out[1]
            boxes, _ = MTCNNUtil.genrate_bb(out0[0, :, :, 1].copy(),
                                           out1[0, :, :, :].copy(),
                                           scale,
                                           threshold[0])

            # inter-scale nms
            pick = nms(boxes.copy(), 0.5, 'Union')
            if boxes.size > 0 and pick.size > 0:
                boxes = boxes[pick, :]
                total_boxes = np.append(total_boxes, boxes, axis=0)

        numbox = total_boxes.shape[0]
        if numbox > 0:
            pick = nms(total_boxes.copy(), 0.7, 'Union')
            total_boxes = total_boxes[pick, :]
            regw = total_boxes[:, 2] - total_boxes[:, 0]
            regh = total_boxes[:, 3] - total_boxes[:, 1]
            qq1 = total_boxes[:, 0] + total_boxes[:, 5] * regw
            qq2 = total_boxes[:, 1] + total_boxes[:, 6] * regh
            qq3 = total_boxes[:, 2] + total_boxes[:, 7] * regw
            qq4 = total_boxes[:, 3] + total_boxes[:, 8] * regh
            total_boxes = np.transpose(np.vstack([qq1, qq2, qq3, qq4,
                                                  total_boxes[:, 4]]))
            total_boxes = rerec(total_boxes.copy())
            total_boxes[:, 0:4] = np.fix(total_boxes[:, 0:4]).astype(np.int32)
            dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph = pad(
                total_boxes.copy(), w, h)

        numbox = total_boxes.shape[0]
        if numbox > 0:
            # second stage
            tempimg = np.zeros((24, 24, 3, numbox))
            for k in range(0, numbox):
                tmp = np.zeros((int(tmph[k]), int(tmpw[k]), 3))
                tmp[dy[k] - 1:edy[k], dx[k] - 1:edx[k],
                :] = img[y[k] - 1:ey[k], x[k] - 1:ex[k], :]
                if (tmp.shape[0] > 0 and tmp.shape[1] > 0 or
                        tmp.shape[0] == 0 and tmp.shape[1] == 0):
                    tempimg[:, :, :, k] = imresample(tmp, (24, 24))
                else:
                    return np.empty()
            tempimg = (tempimg - 127.5) * 0.0078125
            tempimg1 = np.transpose(tempimg, (3, 0, 1, 2))
            out = rnet(tempimg1)
            out0 = np.transpose(out[0])
            out1 = np.transpose(out[1])
            score = out0[1, :]
            ipass = np.where(score > threshold[1])
            total_boxes = np.hstack([total_boxes[ipass[0], 0:4].copy(),
                                     np.expand_dims(score[ipass].copy(), 1)])
            mv = out1[:, ipass[0]]
            if total_boxes.shape[0] > 0:
                pick = nms(total_boxes, 0.5, 'Union')
                total_boxes = total_boxes[pick, :]
                total_boxes = bbreg(total_boxes.copy(), np.transpose(mv[:, pick]))
        return total_boxes




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
