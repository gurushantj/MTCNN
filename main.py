from ONet import ONet
from PNet import PNet
import cv2
import numpy as np

from RNet import RNet
from mtcnn_util.mtcnn_util import MTCNNUtil


class MTCNNMain:
    def __init__(self,img_path):
        pnet = PNet()
        pnet.buildModel()
        weight_data = MTCNNUtil.loadWeights("weights/mtcnn_pnet.npy")
        MTCNNUtil.setWeights(weight_data,pnet.model)
        self.pnet_model = pnet

        rnet = RNet()
        rnet.build_model()
        weight_data = MTCNNUtil.loadWeights("weights/mtcnn_rnet.npy")
        MTCNNUtil.setWeights(weight_data, rnet.model)
        self.rnet_model = rnet

        onet = ONet()
        onet.build_model()
        weight_data = MTCNNUtil.loadWeights("weights/mtcnn_onet.npy")
        MTCNNUtil.setWeights(weight_data, onet.model)
        self.onet_model = onet
        self.img_path = img_path

    def detect_faces(self,minsize=20,factor=0.7):
        img = cv2.imread(self.img_path)
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

        #### First stage started ####
        for scale in scales:
            hs = int(np.ceil(h * scale))
            ws = int(np.ceil(w * scale))
            im_data = cv2.resize(img, (ws, hs), interpolation=cv2.INTER_AREA)
            im_data = (im_data - 127.5) * (1. / 128.0)
            img_x = np.expand_dims(im_data, 0)
            out = self.pnet_model.model.predict(img_x)
            out0 = out[0]
            tmp = np.transpose(out0[0, :, :, 1])
            boxes = MTCNNUtil.genrate_bb(out,threshold=0.8,scale=scale)
            pick_indexes = MTCNNUtil.nms(boxes,threshold=0.5)
            if pick_indexes.size > 0:
                boxes = boxes[pick_indexes,:]
                total_boxes=np.append(total_boxes,boxes,axis=0)
        #### First stage ended ####
        #### Second stage started ####
        numbox = total_boxes.shape[0]
        if numbox > 0:
            pick = MTCNNUtil.nms(total_boxes.copy(), 0.7)
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
            dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph = MTCNNUtil.pad(total_boxes.copy(), w, h)

        numbox = total_boxes.shape[0]
        if numbox > 0:
            tempimg = np.zeros((24, 24, 3, numbox))
            for k in range(0, numbox):
                tmp = np.zeros((int(tmph[k]), int(tmpw[k]), 3))
                tmp[dy[k] - 1:edy[k], dx[k] - 1:edx[k],
                :] = img[y[k] - 1:ey[k], x[k] - 1:ex[k], :]
                if (tmp.shape[0] > 0 and tmp.shape[1] > 0 or
                        tmp.shape[0] == 0 and tmp.shape[1] == 0):
                    tempimg[:, :, :, k] = cv2.resize(tmp, (24, 24))
                else:
                    return np.empty()
            tempimg = (tempimg - 127.5) * 0.0078125
            tempimg1 = np.transpose(tempimg, (3, 0, 1, 2))
            out = self.rnet_model.model.predict(tempimg1)
            out0 = np.transpose(out[0])
            out1 = np.transpose(out[1])
            score = out0[1, :]
            ipass = np.where(score > 0.8)
            total_boxes = np.hstack([total_boxes[ipass[0], 0:4].copy(),
                                     np.expand_dims(score[ipass].copy(), 1)])
            mv = out1[:, ipass[0]]
            if total_boxes.shape[0] > 0:
                pick = MTCNNUtil.nms(total_boxes, 0.7)
                total_boxes = total_boxes[pick, :]
                total_boxes = MTCNNUtil.bbreg(total_boxes.copy(), np.transpose(mv[:, pick]))
                # total_boxes = rerec(total_boxes.copy())

        #Second stage ended#
        numbox = total_boxes.shape[0]
        if numbox > 0:
            # third stage
            total_boxes = np.fix(total_boxes).astype(np.int32)
            dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph = MTCNNUtil.pad(
                total_boxes.copy(), w, h)
            tempimg = np.zeros((48, 48, 3, numbox))
            for k in range(0, numbox):
                tmp = np.zeros((int(tmph[k]), int(tmpw[k]), 3))
                tmp[dy[k] - 1:edy[k], dx[k] - 1:edx[k],
                :] = img[y[k] - 1:ey[k], x[k] - 1:ex[k], :]
                if (tmp.shape[0] > 0 and tmp.shape[1] > 0 or
                        tmp.shape[0] == 0 and tmp.shape[1] == 0):
                    tempimg[:, :, :, k] = cv2.resize(tmp, (48, 48))
                else:
                    return np.empty()
            tempimg = (tempimg - 127.5) * 0.0078125
            tempimg1 = np.transpose(tempimg, (3, 0, 1, 2))
            out = self.onet_model.model.predict(tempimg1)
            out0 = np.transpose(out[0])
            out1 = np.transpose(out[1])
            score = out0[1, :]
            ipass = np.where(score > 0.8)
            total_boxes = np.hstack([total_boxes[ipass[0], 0:4].copy(),
                                     np.expand_dims(score[ipass].copy(), 1)])
            mv = out1[:, ipass[0]]
            if total_boxes.shape[0] > 0:
                total_boxes = MTCNNUtil.bbreg(total_boxes.copy(), np.transpose(mv))
                pick = MTCNNUtil.nms(total_boxes.copy(), 0.7, 'MIN')
                total_boxes = total_boxes[pick, :]
            return total_boxes


    def draw_square(self,rectangle_list,save_path="."):
        img = cv2.imread(self.img_path)
        for rectangle in rectangle_list:
            # cv2.putText(img, str(rectangle[4]),
            #             (int(rectangle[0]), int(rectangle[1])),
            #             cv2.FONT_HERSHEY_SIMPLEX,
            #             0.5, (0, 255, 0))
            cv2.rectangle(img, (int(rectangle[0]), int(rectangle[1])),
                          (int(rectangle[2]), int(rectangle[3])),
                          (255, 0, 0), 2)
        cv2.imwrite(save_path, img)


    def train_pnet(self):
        self.pnet_model = PNet()
        self.pnet_model.buildModel()
        #self.pnet_model.model.load_weights("/mnt/disks/sdb/MTCNN/2000/12")
        #weight_data = MTCNNUtil.loadWeights("initial_weights/initial_weight_pnet.npy")
        #MTCNNUtil.setWeights(weight_data,pnet.model,use_dict=False)
        path = "/mnt/disks/sdb/MTCNN/12/{0}"
        dataset_cls = self.pnet_model.readRecordDataSet(path=[
                                                path.format("dataset_cls_12_0.tf"),
                                                path.format("dataset_cls_12_1.tf"),
                                                path.format("dataset_cls_12_2.tf"),
                                                path.format("dataset_cls_12_3.tf"),
                                                path.format("dataset_cls_12_4.tf"),
                                                path.format("dataset_cls_12_5.tf")
                                                ])
        self.pnet_model.train(10000,dataset_cls,dataset_cls)


m = MTCNNMain("/Users/gurushant/Desktop/modi_cabinet.jpg")
# boxes = m.detect_faces()
# m.draw_square(boxes,save_path="/Users/gurushant/Desktop/modi_cabinet_result.jpg")
m.train_pnet()
