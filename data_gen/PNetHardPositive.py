import os

from Constants import TRAINING_DATA_SOURCE_PATH
import numpy as np
import cv2

def main():
    net = 12
    anno_file = 'wider_face_train.txt'
    save_dir = '/Users/gurushant/Downloads/ds/native_' + str(net)
    pos_save_dir = save_dir + '/positive'
    pos_path = save_dir+"/pos_{0}.txt".format(net)
    file = open(pos_path,"a")
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if not os.path.exists(pos_save_dir):
        os.mkdir(pos_save_dir)

    with open(anno_file, 'r') as f:
        annotations = f.readlines()
    index = 0
    for annotation in annotations:
        annotation = annotation.split()
        img_path = TRAINING_DATA_SOURCE_PATH+annotation[0]+".jpg"
        img = cv2.imread(img_path)
        img = cv2.resize(img,(net,net),interpolation=cv2.INTER_LINEAR)
        offset_x1 = np.random.uniform(-0.1,0.1,size=1)[0]
        offset_y1 = np.random.uniform(-0.1,0.1,size=1)[0]
        offset_x2 = np.random.uniform(-0.1,0.1,size=1)[0]
        offset_y2 = np.random.uniform(-0.1,0.1,size=1)[0]
        path = pos_save_dir + "/hard_pos_{0}.jpg".format(index)
        file.write(path +
                 ' 1 %.2f %.2f %.2f %.2f\n' %
                 (offset_x1, offset_y1, offset_x2, offset_y2))
        cv2.imwrite(path,img)
        index +=1
        file.flush()

    file.close()

main()