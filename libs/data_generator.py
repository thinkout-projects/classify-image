#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 訓練用データ(= trainフォルダ)の準備

import os
import numpy as np
import cv2
import random
import threading
import shutil
import dataclasses
from .utils.utils import fpath_tag_making, fpath_making, read_img, printWithDate
from .utils.folder import folder_create
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm



# 画像データの増強・水増しをopencvのエフェクトを使って行う
@dataclasses.dataclass
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

    data_augment関数内における処理の呼び出し
        data = Trans()
        処理後画像 = data.convert(処理前画像, 処理に対応する数字)
    '''

    contrast: bool = False # コントラスト変換
    gamma: bool = False # ガンマ変換
    blur: bool = False # 平滑化
    equalize_histogram: bool = False # ヒストグラム平坦化
    noise: bool = False # ノイズ


    def __post_init__(self):
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

        self.func_dict = {'nothing': self.nothing_func,
                          'high_contrast': self.hc1_func,
                          'low_contrast': self.lc1_func,
                          'gamma_1': self.gamma1_func,
                          'gamma_2': self.gamma2_func,
                          'blur': self.blur_func,
                          'equalize_histogram': self.equalizeHistRGB_func,
                          'gaussian_noise': self.addGaussianNoise_func,
                          'salt_pepper_noise': self.addSaltPepperNoise_func}

        # 変換の確率
        self.convert_list_1 = []
        if self.contrast:
            self.convert_list_1.append('high_contrast')
            self.convert_list_1.append('low_contrast')
        if self.gamma:
            self.convert_list_1.append('gamma_1')
            self.convert_list_1.append('gamma_2')
        if self.blur:
            self.convert_list_1.append('blur')
        if self.equalize_histogram:
            self.convert_list_1.append('equalize_histogram')
        if len(self.convert_list_1) == 0:
            self.convert_list_1.append('nothing')
        else:
            self.convert_list_1 += len(self.convert_list_1) * ['nothing']

        self.convert_list_2 = []
        if self.noise:
            self.convert_list_2.append('gaussian_noise')
            self.convert_list_2.append('salt_pepper_noise')
        if len(self.convert_list_2) == 0:
            self.convert_list_2.append('nothing')
        else:
            self.convert_list_2 += len(self.convert_list_2) * ['nothing']


    def augment(self):
        def fit(self, input_image):
            input_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)

            rand = np.random.randint(len(self.convert_list_1))
            input_image = self.func_dict[self.convert_list_1[rand]]
            rand = np.random.randint(len(self.convert_list_2))
            input_image = self.func_dict[self.convert_list_2[rand]]

            return input_image


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
        out[tuple(coords[:-1])] = (255, 255, 255)

        # Pepper mode
        num_pepper = np.ceil(amount * src.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i-1, int(num_pepper))
                  for i in src.shape]
        out[tuple(coords[:-1])] = (0, 0, 0)
        return out
