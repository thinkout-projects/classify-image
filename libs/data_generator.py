#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 訓練用データ(= trainフォルダ)の準備

import os
import pandas as pd
import numpy as np
import cv2
import random
import threading
import shutil
from .utils.utils import fpath_tag_making, read_img, printWithDate
from .utils.folder import folder_create
from keras.preprocessing.image import ImageDataGenerator
# from keras.utils import np_utils
from tqdm import tqdm

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
    def __init__(self, folder_names, idx,
                 pic_mode, train_num_mode_dic, size, classes, args_of_IDG, BATCH_SIZE):
        self.source_folder = folder_names['dataset']
        self.dataset_folder = folder_names['split_info']
        self.train_root = folder_names['train']
        self.idx = idx
        self.pic_mode = pic_mode
        self.train_num_mode_dic = train_num_mode_dic
        self.size = size
        self.h = self.size[0]
        self.w = self.size[1]
        self.classes = classes
        self.rotation_range = int(args_of_IDG['rotation_range'])
        self.width_shift_range = float(args_of_IDG['width_shift_range'])
        self.height_shift_range = float(args_of_IDG['height_shift_range'])
        self.shear_range = int(args_of_IDG['shear_range'])
        self.zoom_range = float(args_of_IDG['zoom_range'])
        self.BATCH_SIZE = BATCH_SIZE

    def pic_df_training(self):
        '''
        Datasetから訓練用フォルダを作成
        '''
        df_train = pd.read_csv(os.path.join(
            self.dataset_folder, "train" + "_" + str(self.idx) + "." + "csv"),
            encoding="utf-8")
        columns = df_train.columns

        # train作成
        folder_create(self.train_root)
        for column in columns:
            # train/00_normal作成
            train_folder = os.path.join(self.train_root, column)
            folder_create(train_folder)
            train_list = df_train[column].dropna()

            with tqdm(total=len(train_list),
                      desc='for ' + column, leave=True) as pbar:
                # 画像ごとに
                for train_file in train_list:
                    # img/00_normal/画像
                    img_path = os.path.join(self.source_folder, train_file)
                    src0 = cv2.imread(img_path)
                    file_without = train_file.split(".")[0]

                    # train_num_mode_dicはフォルダ名をKeyとして、numとmodeのリストを保持している。
                    # num・・・9個の変換の中で指定の数だけを作成する。
                    # mode・・・0なら左右反転なし、1なら左右反転あり。
                    num_mode = self.train_num_mode_dic[column]
                    num = num_mode[0]
                    mode = num_mode[1]
                    num_list = random.sample(range(9), num)
                    data_augment(train_folder, file_without,
                                 src0, num_list, mode)
                    pbar.update(1)
        return

    def data_gen(self, X_train):
        '''
        さらにnumpyの中でデータ拡張を行うもの
        ImageDataGeneratorは設定した後、`fit`して使うもの
        `fit`の引数は`X_train`
        代入しなくても、`fit`するだけでrandomに`X_train`の中身をいじってくる。
        '''
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

    def mini_batch(self, fpath_list, tag_array, i):
        X_train = []
        y_train = []
        for b in range(self.BATCH_SIZE):
            fpath = fpath_list[i + b]
            X = read_img(fpath, self.h, self.w)
            X_train.append(X)
            # 回帰の場合はファイル名冒頭からターゲットを読み込む
            if(self.pic_mode == 2):
                y_train.append(int(fpath.split("\\")[-1].split("_")[2]))
        X_train = np.array(X_train)
        X_train = Training.data_gen(self, X_train)
        if(self.pic_mode != 2):
            y_train = tag_array[i: i + self.BATCH_SIZE]
        else:
            y_train = np.array(y_train)
        return X_train, y_train

    @threadsafe_generator
    def datagen(self, fpath_list, tag_array):  # data generator
        while True:
            for i in range(0, len(fpath_list) - self.BATCH_SIZE,
                           self.BATCH_SIZE):
                x, t = Training.mini_batch(self, fpath_list, tag_array, i)
                if(t[0].size == self.classes):  # 謎バグ回避用
                    yield x, t
                else:
                    i -= self.BATCH_SIZE


class Validation(object):
    '''
    評価用データの作成および読み込みのクラス
    '''

    def __init__(self, size, folder_names, classes, pic_mode, idx):
        self.source_folder = folder_names['dataset']
        self.dataset_folder = folder_names['split_info']
        self.test_root = folder_names['test']
        self.idx = idx
        self.pic_mode = pic_mode
        self.classes = classes
        self.size = size
        self.h = self.size[0]
        self.w = self.size[1]
        self.random_seed = 1

    def pic_df_test(self):
        '''
        評価用データの画像を出力（画像解析用、カテゴリー分類）
        '''

        df_test = pd.read_csv(os.path.join(
            self.dataset_folder, "test" + "_" + str(self.idx) + "." + "csv"),
            encoding="utf-8")
        columns = df_test.columns

        # test作成
        folder_create(self.test_root)
        for column in columns:
            # test/00_Normal作成
            test_folder = os.path.join(self.test_root, column)
            folder_create(test_folder)
            test_list = df_test[column].dropna()

            with tqdm(total=len(test_list),
                      desc='for ' + column, leave=True) as pbar:
                for test_file in test_list:
                    # img/ 00_normal/ filename
                    img_path = os.path.join(
                        self.source_folder, test_file)
                    # test/00_normal/filename
                    new_path = os.path.join(test_folder, test_file)
                    shutil.copy(img_path, new_path)
                    pbar.update(1)
        return

    def pic_gen_data(self):
        '''
        評価用データ生成（画像からtest/00_normal/画像となっている想定）
        '''

        fpath_list, tag_array = fpath_tag_making(self.test_root, self.classes)
        X_val = []
        y_val = []
        for fpath in fpath_list:
            X = read_img(fpath, self.h, self.w)
            X_val.append(X)
            if(self.pic_mode == 2):
                y_val.append(int(fpath.split("\\")[-1].split("_")[0]))

        # 全てを再度array化する
        X_val = np.array(X_val)
        if(self.pic_mode == 2):
            y_val = np.array(y_val)

        # class数, validation dataがいくつか
        printWithDate(len(X_val), " files for validation")
        # validationのデータとlabel、ファイルパス
        if(self.pic_mode != 2):
            return (X_val, tag_array, fpath_list)
        else:
            return (X_val, y_val, fpath_list)


# 画像データの増強・水増しをopencvのエフェクトを使って行う

def data_augment(newfolder, file, src0, num_list, mode):
    '''
    `newfolder`は保存先のフォルダ
    `file`は元々のfile名（拡張子なし）→のちにjpgをたす
    `src0`は元の画像（numpy array型式）
    `num_list`はどの変換を適応するか
    `mode`はflipのどれを含めるか（2018/4/1時点では左右flipのみ）
    '''

    data = Trans()
    for num in num_list:
        if mode == 0:
            dst0 = data.convert(src0, num)
            newfile0 = str(num) + "_" + file + ".jpg"
            newfpath0 = os.path.join(newfolder, newfile0)
            # 全ての画像に同じ処理をしたい場合はここに！マスキングなど
            cv2.imwrite(newfpath0, dst0)
        if mode == 1:
            src1 = data.lr_flip_func(src0)
            dst0 = data.convert(src0, num)
            dst1 = data.convert(src1, num)
            newfile0 = "0" + "_" + str(num) + "_" + file + ".jpg"
            newfile1 = "1" + "_" + str(num) + "_" + file + ".jpg"
            newfpath0 = os.path.join(newfolder, newfile0)
            newfpath1 = os.path.join(newfolder, newfile1)
            cv2.imwrite(newfpath0, dst0)
            cv2.imwrite(newfpath1, dst1)


class Trans:
    '''
    Transクラスで具体的な処理を記述
        ①そのまま、 nothing
        ②High contrast  high_contrast1
        ③Low contrast   low_contrast1
        ④Gamma変換1     gamma1
        ⑤Gamma変換2     gamma2
        ⑥平滑化         blur
        ⑦ヒストグラム均一化     equalizeHistRGB
        ⑧ガウシアンノイズ       addGaussianNoise
        ⑨ソルトペッパーノイズ   addSaltPepperNoise
        ⑩左右反転               lr_flip_func

    data_augment関数内における処理の呼び出し
        data = Trans()
        処理後画像 = data.convert(処理前画像, 処理に対応する数字)
    '''

    def __init__(self):
        # ルックアップテーブルの生成
        self.min_table = 50
        self.max_table = 205
        self.diff_table = self.max_table - self.min_table
        self.gamma1 = 0.75
        self.gamma2 = 1.5

        # 平滑化用
        self.average_square = (10, 10)

        self.LUT_HC = np.arange(256, dtype='uint8')
        self.LUT_LC = np.arange(256, dtype='uint8')
        self.LUT_G1 = np.arange(256, dtype='uint8')
        self.LUT_G2 = np.arange(256, dtype='uint8')
        self.LUTs = []

        # ハイコントラストLUT作成
        for i in range(0, self.min_table):
            self.LUT_HC[i] = 0

        for i in range(self.min_table, self.max_table):
            self.LUT_HC[i] = 255 * (i - self.min_table) / self.diff_table

        for i in range(self.max_table, 255):
            self.LUT_HC[i] = 255

        # その他LUT作成
        for i in range(256):
            self.LUT_LC[i] = self.min_table + i * (self.diff_table) / 255
            self.LUT_G1[i] = 255 * pow(float(i) / 255, 1.0 / self.gamma1)
            self.LUT_G2[i] = 255 * pow(float(i) / 255, 1.0 / self.gamma2)

        self.LUTs.append(self.LUT_HC)
        self.LUTs.append(self.LUT_LC)
        self.LUTs.append(self.LUT_G1)
        self.LUTs.append(self.LUT_G2)

        self.func_list = [self.nothing_func, self.hc1_func, self.lc1_func,
                          self.gamma1_func, self.gamma2_func, self.blur_func,
                          self.equalizeHistRGB_func,
                          self.addGaussianNoise_func,
                          self.addSaltPepperNoise_func, self.lr_flip_func]

    # ① 何もしない
    def nothing_func(self, src):
        return src

    # ② high contrast
    def hc1_func(self, src):
        img_dst = cv2.LUT(src, self.LUTs[0])
        return img_dst

    # ③ low_contrast1
    def lc1_func(self, src):
        img_dst = cv2.LUT(src, self.LUTs[1])
        return img_dst

    # ④ gamma補正1
    def gamma1_func(self, src):
        img_dst = cv2.LUT(src, self.LUTs[2])
        return img_dst

    # ⑤ gamma補正2
    def gamma2_func(self, src):
        img_dst = cv2.LUT(src, self.LUTs[3])
        return img_dst

    # ⑥ 平滑化
    def blur_func(self, src):
        img_dst = cv2.blur(src, self.average_square)
        return img_dst

    # ⑦ ヒストグラム均一化
    def equalizeHistRGB_func(self, src):
        RGB = cv2.split(src)
        # Blue = RGB[0]
        # Green = RGB[1]
        # Red = RGB[2]
        for i in range(3):
            cv2.equalizeHist(RGB[i])
        img_hist = cv2.merge([RGB[0], RGB[1], RGB[2]])
        return img_hist

    # ⑧ ガウシアンノイズ
    def addGaussianNoise_func(self, src):
        row, col, ch = src.shape
        mean = 0
        # var = 0.1
        sigma = 15
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = src + gauss
        return noisy

    # ⑨ salt&pepperノイズ
    def addSaltPepperNoise_func(self, src):
        row, col, ch = src.shape
        s_vs_p = 0.5
        amount = 0.004
        out = src.copy()

        # Salt mode
        num_salt = np.ceil(amount * src.size * s_vs_p)
        coords = [np.random.randint(0, i-1, int(num_salt))
                  for i in src.shape]
        out[coords[:-1]] = (255, 255, 255)

        # Pepper mode
        num_pepper = np.ceil(amount * src.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i-1, int(num_pepper))
                  for i in src.shape]
        out[coords[:-1]] = (0, 0, 0)
        return out

    # ⑩ 反対
    def lr_flip_func(self, src):
        img_dst = cv2.flip(src, 1)
        return img_dst

    def convert(self, src, num):
        img_dst = self.func_list[num](src)
        return img_dst
