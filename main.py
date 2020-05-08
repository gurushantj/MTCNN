from Constants import TRAINING_DATA_SOURCE_PATH, DATASET_SAVE_DIR
from ONet import ONet
from PNet import PNet
import cv2
import numpy as np
import tensorflow as tf
from RNet import RNet
from mtcnn_util.mtcnn_util import MTCNNUtil, Mode
import os

class MTCNNMain:
    def __init__(self,img_path,mode):
        pnet = PNet()
        pnet.buildModel()
        self.pnet_model = pnet

        rnet = RNet()
        rnet.build_model()
        self.rnet_model = rnet

        onet = ONet()
        onet.build_model()

        self.onet_model = onet
        self.img_path = img_path

        if mode == Mode.TESTING.value:
            weight_data = MTCNNUtil.loadWeights("weights/mtcnn_pnet.npy")
            # MTCNNUtil.setWeights(weight_data,pnet.model)
            pnet.model = tf.keras.models.load_model("/Users/gurushant/model/2000/12")

            weight_data = MTCNNUtil.loadWeights("weights/mtcnn_rnet.npy")
            MTCNNUtil.setWeights(weight_data, rnet.model)
            weight_data = MTCNNUtil.loadWeights("weights/mtcnn_onet.npy")
            MTCNNUtil.setWeights(weight_data, onet.model)


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
            ipass = np.where(score >= 0.8)
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

    def preprocess_image(self,im_data):
        return (im_data - 127.5) * (1. / 128.0)

    def generate_hard_12(self):
        threshold = 0.6
        minsize = 20
        factor = 0.709
        self.pnet_model.model = tf.keras.models.load_model("/Users/gurushant/model/2000/12")
        image_size = 24
        save_dir = DATASET_SAVE_DIR+str(image_size)
        anno_file = 'wider_face_train.txt'

        neg_save_dir = save_dir + '/negative'
        pos_save_dir = save_dir + '/positive'
        part_save_dir = save_dir + '/part'
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        if not os.path.exists(pos_save_dir):
            os.mkdir(pos_save_dir)
        if not os.path.exists(part_save_dir):
            os.mkdir(part_save_dir)
        if not os.path.exists(neg_save_dir):
            os.mkdir(neg_save_dir)

        f1 = open(save_dir + '/pos_24.txt', 'w')
        f2 = open(save_dir + '/neg_24.txt', 'w')
        f3 = open(save_dir + '/part_24.txt', 'w')

        with open(anno_file, 'r') as f:
            annotations = f.readlines()
        num = len(annotations)
        print('%d pics in total' % num)

        p_idx = 0  # positive
        n_idx = 0  # negative
        d_idx = 0  # dont care
        image_idx = 0


        model = self.pnet_model.model
        with open("wider_face_train.txt","r") as file:
            annotation_lines = file.readlines()

        def pnet(img):
            return model.predict(img)

        for line in annotation_lines:
            line = line.split()
            img_path = TRAINING_DATA_SOURCE_PATH+line[0]+".jpg"
            img = cv2.imread(img_path)
            rects = MTCNNUtil.detect_face_12net(img,minsize,pnet,threshold,factor)
            boxes = np.reshape(line[1:],newshape=(-1,4))
            boxes = boxes.astype(np.float64)
            # cv2.rectangle(img,(452,421),(468,440),(0,0,255),2)
            # cv2.rectangle(img,(448,329),(570,478),(0,255,0),2)
            # cv2.imshow("test",img)
            # cv2.waitKey()
            for box in rects:
                box[box < 0] = 0
                box = box.astype(np.int32)
                # print(box[2]-box[0])
                # print(box[3]-box[1])
                if (box[2]-box[0]) < 15 or (box[3]-box[1]) < 15:
                    continue
                x_left, y_top, x_right, y_bottom, _ = box
                crop_w = x_right - x_left + 1
                crop_h = y_bottom - y_top + 1
                # ignore box that is too small or beyond image border
                if crop_w < image_size or crop_h < image_size:
                    continue

                iou = MTCNNUtil.iou(box,boxes)
                cropped_im = img[y_top: y_bottom + 1, x_left: x_right + 1]
                resized_im = cv2.resize(cropped_im,
                                        (image_size, image_size),
                                        interpolation=cv2.INTER_LINEAR)
                # save negative images and write label
                if np.max(iou) < 0.3:
                    # Iou with all gts must below 0.3
                    save_file = os.path.join(neg_save_dir,
                                             '%s.jpg' % n_idx)
                    f2.write('%s/negative/%s' %
                             (save_dir, n_idx) + ' 0\n')
                    cv2.imwrite(save_file, resized_im)
                    n_idx += 1
                else:
                    # find gt_box with the highest iou
                    idx = np.argmax(iou)
                    assigned_gt = boxes[idx]
                    x1, y1, x2, y2 = assigned_gt

                    # compute bbox reg label
                    offset_x1 = (x1 - x_left) / float(crop_w)
                    offset_y1 = (y1 - y_top) / float(crop_h)
                    offset_x2 = (x2 - x_right) / float(crop_w)
                    offset_y2 = (y2 - y_bottom) / float(crop_h)

                    if np.max(iou) >= 0.65:
                        save_file = os.path.join(pos_save_dir,
                                                 '%s.jpg' % p_idx)
                        f1.write('%s/positive/%s' % (save_dir, p_idx) +
                                 ' 1 %.2f %.2f %.2f %.2f\n' %
                                 (offset_x1, offset_y1,
                                  offset_x2, offset_y2))
                        cv2.imwrite(save_file, resized_im)
                        p_idx += 1

                    elif np.max(iou) >= 0.4:
                        save_file = os.path.join(part_save_dir,
                                                 '%s.jpg' % d_idx)
                        f3.write('%s/part/%s' % (save_dir, d_idx) +
                                 ' -1 %.2f %.2f %.2f %.2f\n' %
                                 (offset_x1, offset_y1,
                                  offset_x2, offset_y2))
                        cv2.imwrite(save_file, resized_im)
                        d_idx += 1


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


    def train_pnet(self,path):
        #self.pnet_model.model.load_weights("/mnt/disks/sdb/MTCNN/2000/12")
        #weight_data = MTCNNUtil.loadWeights("initial_weights/initial_weight_pnet.npy")
        #MTCNNUtil.setWeights(weight_data,pnet.model,use_dict=False)
        self.pnet_model.model.load_model("/Users/gurushant/model/9000/12")
        file_list = MTCNNUtil.get_files(path)
        dataset_cls = self.pnet_model.readRecordDataSet(path=file_list)
        self.pnet_model.train(10000,dataset_cls,dataset_cls)


    def train_rnet(self,path):
        #self.pnet_model.model.load_weights("/mnt/disks/sdb/MTCNN/2000/12")
        #weight_data = MTCNNUtil.loadWeights("initial_weights/initial_weight_pnet.npy")
        #MTCNNUtil.setWeights(weight_data,pnet.model,use_dict=False)
        file_list = MTCNNUtil.get_files(path)
        dataset_cls = self.rnet_model.readRecordDataSet(path=file_list)
        self.rnet_model.train(10000,dataset_cls)


    def train_onet(self,path):
        #self.pnet_model.model.load_weights("/mnt/disks/sdb/MTCNN/2000/12")
        #weight_data = MTCNNUtil.loadWeights("initial_weights/initial_weight_pnet.npy")
        #MTCNNUtil.setWeights(weight_data,pnet.model,use_dict=False)
        file_list = MTCNNUtil.get_files(path)
        dataset_cls = self.onet_model.readRecordDataSet(path=file_list)
        self.onet_model.train(10000,dataset_cls)




path = "/Users/gurushant/Downloads/WIDER_train/images/0--Parade/0_Parade_marchingband_1_849.jpg"
m = MTCNNMain(path,mode=Mode.TESTING.value)
m.generate_hard_12()
# boxes = m.detect_faces()
# if boxes.size == 0 :
#     print("No face found")
#     exit(100)
# m.draw_square(boxes,save_path="/Users/gurushant/Desktop/band.jpg")
# m.train_pnet("/mnt/disks/sdb/MTCNN/48")
