#! env python
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import cv2
import random
import threading
from data_augment import data_augment
from utils import folder_create, fpath_tag_making,read_img
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils



# Projects.pic_data_augment
# Date: 2018/04/02
# Filename: pic_data_augment 
# To change this template, choose Tools | Templates
# and open the template in the editor.
__author__ = 'masuo'
__date__ = "2018/04/02"

# k_fold_split(k,img_root,dataset_folder)
# train_0.csvなどとなる。columnsはimg_rootにあるフォルダ名
# data_augment(newfolder
# newfolderは保存先
# file名（拡張子なし）→保存先でのファイル名のために
# src →numpy array
# num_listは変換
# mode

class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return next(self.it)


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g


class Training(object):
    def __init__(self, source_folder,dataset_folder, train_root,idx,pic_mode,train_num_mode_dic,size,classes,
                 rotation_range, width_shift_range, height_shift_range, shear_range, zoom_range,BATCH_SIZE):
        self.source_folder = source_folder
        self.train_root = train_root
        self.idx = idx
        self.pic_mode = pic_mode
        self.dataset_folder = dataset_folder
        self.train_num_mode_dic = train_num_mode_dic
        self.size = size
        self.h = self.size[0]
        self.w = self.size[1]
        self.classes = classes
        self.rotation_range = rotation_range
        self.width_shift_range = width_shift_range
        self.height_shift_range = height_shift_range
        self.shear_range = shear_range
        self.zoom_range = zoom_range
        self.BATCH_SIZE = BATCH_SIZE

    # Datasetから訓練用フォルダを作成
    def pic_df_training(self):
        df_train = pd.read_csv(os.path.join(self.dataset_folder, "train" + "_" + str(self.idx) + "." + "csv"), encoding="utf-8")
        columns = df_train.columns
        # train作成
        folder_create(self.train_root)
        for column in columns:
            # train/00_normal作成
            train_folder = os.path.join(self.train_root, column)
            folder_create(train_folder)
            train_list = df_train[column].dropna()
            # 画像ごとに
            for train_file in train_list:
                print(train_file)
                # img/00_normal/画像
                img_path = os.path.join(self.source_folder, column, train_file)
                src0 = cv2.imread(img_path)
                file_without = train_file.split(".")[0]
                # train_num_mode_dicはフォルダ名をKeyとして、numとmodeのリストを保持している。
                # num・・・9個の変換の中で指定の数だけを作成する。
                # mode・・・0なら左右反転なし、1なら左右反転あり。
                num_mode = self.train_num_mode_dic[column]
                num = num_mode[0]
                mode = num_mode[1]
                num_list = random.sample(range(9), num)
                data_augment(train_folder, file_without, src0, num_list, mode)
        return


    # さらにnumpyの中でデータ拡張を行うもの。
    def data_gen(self,X_train):
        # ImageDataGeneratorは設定した後、fitして使うもの
        # fitの引数はX_train
        # 代入しなくても、fitするだけでrandomにX_trainの中身をいじってくる。
        IDG = ImageDataGenerator(
            # rescale=1./255,
            rotation_range=self.rotation_range,
            width_shift_range=self.width_shift_range,
            height_shift_range=self.height_shift_range,
            shear_range=self.shear_range,
            zoom_range=self.zoom_range,
            horizontal_flip=False,
            vertical_flip=False,
        )
        IDG.fit(X_train)
        return X_train


    def mini_batch(self,fpath_list,tag_array,i):
        X_train = []
        y_train = []
        for b in range(self.BATCH_SIZE):
            fpath = fpath_list[i +b]
            X = read_img(fpath, self.h, self.w)
            X_train.append(X)
            # 回帰の場合はファイル名冒頭からターゲットを読み込む
            if(self.pic_mode == 2): y_train.append(int(fpath.split("\\")[-1].split("_")[2]))
        X_train = np.array(X_train)
        X_train = Training.data_gen(self,X_train)
        if(self.pic_mode != 2): y_train = tag_array[i: i+ self.BATCH_SIZE]
        else: y_train = np.array(y_train)
        return X_train, y_train

    @threadsafe_generator
    def datagen(self,fpath_list,tag_array): # data generator
        while True:
            for i in range(0, len(fpath_list) - self.BATCH_SIZE, self.BATCH_SIZE):
                x, t = Training.mini_batch(self,fpath_list,tag_array,i)
                if(t[0].size == self.classes): # 謎バグ回避用
                    yield x, t
                else:
                    i -= self.BATCH_SIZE

