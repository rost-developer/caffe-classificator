import os
import glob
import random
import numpy as np

import cv2

import caffe
from caffe.proto import caffe_pb2
import lmdb


class ImgPreparing(object):
    # Size of images
    IMAGE_WIDTH = 227
    IMAGE_HEIGHT = 227

    def __init__(self, param):
        # self.train_lmdb = param.train_lmdb
        # self.validation_lmdb = param.train_lmdb
        # self.train_data = [img for img in glob.glob(param.train)]
        # self.test_data = [img for img in glob.glob(param.test)]

    # def prepare_path(self):
    #     os.system('rm -rf  ' + self.train_lmdb)
    #     os.system('rm -rf  ' + self.validation_lmdb)

    def transform_img(self, img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT):
        # Histogram Equalization
        img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
        img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
        img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])

        # Image Resizing
        img = cv2.resize(img, (img_width, img_height), interpolation=cv2.INTER_CUBIC)

        return img

    def make_datum(self, img, label):
        # image is numpy.ndarray format. BGR instead of RGB
        return caffe_pb2.Datum(
            channels=3,
            width=self.IMAGE_WIDTH,
            height=self.IMAGE_HEIGHT,
            label=label,
            data=np.rollaxis(img, 2).tostring())

    def create_lmdb(self, data, lmdb_path):
        # Shuffle train_data
        random.shuffle(data)
        print '\nStarting creation lmdb...'
        in_db = lmdb.open(lmdb_path, map_size=int(1e12))
        with in_db.begin(write=True) as in_txn:
            for in_idx, img_path in enumerate(data):
                if in_idx % 6 == 0:
                    continue
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                img = self.transform_img(img, img_width=self.IMAGE_WIDTH, img_height=self.IMAGE_HEIGHT)
                if 'cat' in img_path:
                    label = 0
                else:
                    label = 1
                datum = self.make_datum(img, label)
                in_txn.put('{:0>5d}'.format(in_idx), datum.SerializeToString())
                print '{:0>5d}'.format(in_idx) + ':' + img_path
        in_db.close()
        print '\nFinish creation lmdb'

