#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 評価用データの作成および読みこみ

import os
# import sys
import pandas as pd
import shutil
import numpy as np
# import cv2
# import random
# from keras.utils import np_utils
from utils import folder_create, fpath_tag_making, read_img
from utils import printWithDate
from tqdm import tqdm


class Validation(object):
    '''
    評価用データの作成および読み込みのクラス
    '''

    def __init__(self, size, source_folder, test_root, dataset_folder,
                 classes, pic_mode, idx):
        self.source_folder = source_folder
        self.test_root = test_root
        self.dataset_folder = dataset_folder
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
                        self.source_folder, column, test_file)
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
