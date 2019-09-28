#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 毎回全て書くのが面倒な雑多な処理集

import os
import pandas as pd
import numpy as np
# from scipy import stats
import scipy
import cv2
from keras.utils import np_utils

from pytz import timezone
from datetime import datetime
from .. import error


def printWithDate(*printee):
    '''
    通常のprint文に加え、[ yyyy/mm/dd hh:mm:ss ] を文頭に挿入する。
    ただし、`sep`, `end`, `file`, `flush`は使用不可能。
    '''
    print("[", datetime.now().astimezone(timezone('Asia/Tokyo'))
          .strftime("%Y/%m/%d %H:%M:%S"), "] ", end="")
    for i in printee:
        print(i, end="")
    print()
    return


def check_options(Options):
    '''
    options.confが想定通りのフォーマットになっているかチェックする
    '''
    SECTIONS = ['FolderName',
                'CSV',
                'NetworkUsing',
                'ImageSize',
                'HyperParameter',
                'DataGenerate',
                'ImageDataGenerator',
                'Validation',
                'Analysis',
                'etc']

    OPTIONS = [  # FolderName
               ['dataset',
                'split_info',
                'train',
                'test'],
               # CSV
               ['csv_filename',
                'image_filename_column',
                'ID_column',
                'label_column'],
               # NetworkUsing
               ['VGG16',
                'VGG19',
                'DenseNet121',
                'DenseNet169',
                'DenseNet201',
                'InceptionResNetV2',
                'InceptionV3',
                'ResNet50',
                'Xception',
                'LightWeight'],
               # ImageSize
               ['width',
                'height'],
               # HyperParameter
               ['batch_size',
                'epochs'],
               # DataGenerate
               ['num_of_augs',
                'use_flip'],
               # ImageDataGenerator
               ['rotation_range',
                'width_shift_range',
                'height_shift_range',
                'shear_range',
                'zoom_range'],
               # Validation
               ['k'],
               # Analysis
               ['alpha',
                'positive_label'],
               # etc
               ['wait_sec']]

    # 欠けているセクション・オプションがないか調べる
    for section_id, section_name in enumerate(SECTIONS):
        if Options.has_section(section_name) is False:
            error.section_not_found(section_name)
        else:
            for option_name in OPTIONS[section_id]:
                if Options.has_option(section_name, option_name) is False:
                    error.option_not_found(section_name, option_name)

    return


def ID_reading(dataset_folder, idx):
    df = pd.read_csv(os.path.join(dataset_folder, f"dataset_{idx}.csv"),
                     encoding="utf-8")
    # ファイル名一覧
    train_list = df["train"].dropna()
    test_list = df["test"].dropna()
    return train_list, test_list


def clopper_pearson(k, n, alpha):
    alpha2 = (1 - alpha) / 2
    lower = scipy.stats.beta.ppf(alpha2, k, n - k + 1)
    upper = scipy.stats.beta.ppf(1 - alpha2, k + 1, n - k)
    return (lower, upper)


def split_array(ar, n_group):
    for i_chunk in range(n_group):
        yield ar[i_chunk * len(ar) // n_group:(i_chunk + 1)
                 * len(ar) // n_group]


def list_shuffle(a, b, seed):
    np.random.seed(seed)
    list_ = list(zip(a, b))
    np.random.shuffle(list_)
    a1, b1 = zip(*list_)
    a2 = list(a1)
    b2 = list(b1)
    return a2, b2


def fpath_tag_making(root, classes, positive_label=""):
    '''
    「2クラス分類」か「それ以外」かで処理を分岐する
    '''

    if classes == 2:
        return fpath_tag_making_binary(root, positive_label)
    else:
        return fpath_tag_making_categorical(root, classes)


def fpath_tag_making_binary(root, positive_label):
    '''
    root/00_normal/画像を見に行く。
    filepathのリストとtagのカテゴリー化済みのarrayが出力される。
    '''

    # train
    seed = 1
    file_folder_list = os.listdir(root)
    folder_list = [f for f in file_folder_list
                   if os.path.isdir(os.path.join(root, f))]
    fpath_list = []
    tag_list = []
    # train/00_tgt
    for folder in folder_list:
        if folder == positive_label:
            tag = 1
        else:
            tag = 0
        folder_path = os.path.join(root, folder)
        file_list = os.listdir(folder_path)
        for file_ in file_list:
            fpath = os.path.join(folder_path, file_)
            fpath_list.append(fpath)
            tag_list.append(tag)
    fpath_list, tag_list = list_shuffle(fpath_list, tag_list, seed)
    tag_array = np.array(tag_list)
    tag_array = np_utils.to_categorical(tag_array, 2)
    return fpath_list, tag_array


def fpath_tag_making_categorical(root, classes):
    '''
    root/00_normal/画像を見に行く。
    filepathのリストとtagのカテゴリー化済みのarrayが出力される。
    '''

    # train
    seed = 1
    file_folder_list = os.listdir(root)
    folder_list = [f for f in file_folder_list
                   if os.path.isdir(os.path.join(root, f))]
    fpath_list = []
    tag_list = []
    # train/00_tgt
    for tag, folder in enumerate(folder_list):
        folder_path = os.path.join(root, folder)
        file_list = os.listdir(folder_path)
        for file_ in file_list:
            fpath = os.path.join(folder_path, file_)
            fpath_list.append(fpath)
            tag_list.append(tag)
    fpath_list, tag_list = list_shuffle(fpath_list, tag_list, seed)
    tag_array = np.array(tag_list)
    # OPTIMIZE: classesはroot以下のディレクトリの数を数えて取得出来るので、
    #           引数としてもらわなくても良いのではないか
    tag_array = np_utils.to_categorical(tag_array, classes)
    return fpath_list, tag_array


def fpath_making(root):
    '''
    root/00_normal/画像を見に行く。
    filepathのリストとtagのカテゴリー化済みのarrayが出力される。
    '''

    # train
    seed = 1
    file_folder_list = os.listdir(root)
    folder_list = [f for f in file_folder_list
                   if os.path.isdir(os.path.join(root, f))]
    fpath_list = []
    # train/00_tgt
    for folder in folder_list:
        folder_path = os.path.join(root, folder)
        file_list = os.listdir(folder_path)
        for file_ in file_list:
            fpath = os.path.join(folder_path, file_)
            fpath_list.append(fpath)
    np.random.seed(seed)
    np.random.shuffle(fpath_list)
    return fpath_list


def num_count(root):
    '''
    訓練用データ数のカウント
    '''

    dic = {}
    folder_list = os.listdir(root)
    for folder in folder_list:
        folder_path = os.path.join(root, folder)
        file_list = os.listdir(folder_path)
        dic[folder] = len(file_list)
    return dic


def read_img(fpath, h, w):
    '''
    fpathのファイルをh x wのサイズにリサイズし、numpy配列(float32)にしたものを返す
    '''
    X = np.array(
        cv2.resize(cv2.imread(fpath), (h, w)) / 255.0,
        dtype=np.float32)
    return X
