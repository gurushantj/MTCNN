import tensorflow as tf
import cv2
import os
import numpy as np
import random
import sys

from Constants import DATASET_SAVE_DIR


def createFeature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def createTFRecord(lines,startIndex,rowCount,rel_path,is_cls=True):
    records = []
    print("--------------------------------------------------------------->><<>")
    for j in range(startIndex, rowCount):
        line = lines[j]
        lineArr = line.split()
        imagePath = lineArr[0]
        image = cv2.imread(rel_path + imagePath + ".jpg")
        image = image.astype(np.float32)
        image = (image - 127.5) * (1. / 128.0)
        image = image.tostring()

        boundingBox = None
        lineArr[1] = int(lineArr[1])
        if is_cls:
            if lineArr[1] == 1 :
                originalLabel = np.array([0, 1], dtype=np.float32)
                boundingBox = lineArr[2:]

                boundingBox = np.array([boundingBox[0], boundingBox[1], boundingBox[2], boundingBox[3]])
                boundingBox = boundingBox.astype(np.float32)
                boundingBox = boundingBox.tostring()

            elif lineArr[1] == 0:
                originalLabel = np.array([1, 0], dtype=np.float32)
                boundingBox = np.array([0, 0, 0, 0], dtype=np.float32)
                boundingBox = boundingBox.tostring()

        else:
            originalLabel = np.array([0, 0], dtype=np.float32)
            boundingBox = lineArr[2:]
            boundingBox = np.array([boundingBox[0], boundingBox[1], boundingBox[2], boundingBox[3]])
            boundingBox = boundingBox.astype(np.float32)
            boundingBox = boundingBox.tostring()

        originalLabel = originalLabel.tostring()

        image = tf.train.BytesList(value=[image])
        label = tf.train.BytesList(value=[originalLabel])
        boundingBox = tf.train.BytesList(value=[boundingBox])

        imageFeature = tf.train.Feature(bytes_list=image)
        labelFeature = tf.train.Feature(bytes_list=label)
        bbFeature = tf.train.Feature(bytes_list=boundingBox)

        featuresDict = {"image": imageFeature, "label": labelFeature, "boundingBox": bbFeature}
        features = tf.train.Features(feature=featuresDict)
        example = tf.train.Example(features=features)
        records.append(example)
    return records

if len(sys.argv) < 1:
    print("python3 data_gen/CreateTFRecordTraining.py <part>")

part = sys.argv[1]
rel_path = DATASET_SAVE_DIR.format(part)
rel_path_without_native = ""

os.system("mkdir -p {0}".format(part))
def generate_data_for_cls():
    files = [os.path.join(rel_path,"pos_{0}.txt".format(part)),
             os.path.join(rel_path,"neg_{0}.txt".format(part))]

    record_paths = []
    for file in files:
        with open(file,"r") as file:
            record_paths = record_paths+file.readlines()

    random.shuffle(record_paths)
    random.shuffle(record_paths)
    total_records = len(record_paths)

    per_file_record_count = 100
    rec_index =0
    index = 0
    tf_record_path = "{0}/dataset_cls_{0}.tf".format(part)
    while rec_index < total_records:
        if rec_index+per_file_record_count > total_records:
            end_index = rec_index+per_file_record_count-total_records
            end_index = rec_index+per_file_record_count-end_index
        else:
            end_index = rec_index+per_file_record_count

        tf_records = createTFRecord(record_paths,rec_index,end_index,rel_path_without_native)
        tf_record_path = "{0}/dataset_cls_{1}_{2}.tf".format(part,part,index)
        tfWriter = tf.io.TFRecordWriter(tf_record_path)
        for record in tf_records:
            tfWriter.write(record.SerializeToString())
        tfWriter.close()

        print("start index {0},end index {1},size {2}".format(rec_index,end_index,(end_index-rec_index)))
        rec_index = end_index
        index += 1


def generate_data_for_bb():
    files = [os.path.join(rel_path, "pos_{0}.txt".format(part)),
             os.path.join(rel_path, "part_{0}.txt".format(part))]

    record_paths = []
    for file in files:
        with open(file, "r") as file:
            record_paths = record_paths + file.readlines()

    random.shuffle(record_paths)
    random.shuffle(record_paths)
    total_records = len(record_paths)

    per_file_record_count = 200000
    rec_index = 0
    index = 0
    while rec_index < total_records:
        if rec_index + per_file_record_count > total_records:
            end_index = rec_index + per_file_record_count - total_records
            end_index = rec_index + per_file_record_count - end_index
        else:
            end_index = rec_index + per_file_record_count

        tf_records = createTFRecord(record_paths, rec_index, end_index, rel_path_without_native,is_cls=False)
        tf_record_path = "{0}/dataset_bb_{1}_{2}.tf".format(part,part,index)
        tfWriter = tf.io.TFRecordWriter(tf_record_path)
        for record in tf_records:
            tfWriter.write(record.SerializeToString())
        tfWriter.close()

        print("start index {0},end index {1},size {2}".format(rec_index, end_index, (end_index - rec_index)))
        rec_index = end_index
        index += 1

generate_data_for_cls()