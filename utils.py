#! env python
# -*- coding: utf-8 -*-

import os
import shutil
import pandas as pd
import numpy as np
from scipy import stats
import scipy
import cv2
import glob
from keras.utils import np_utils


# Projects.utils
# Date: 2018/04/02
# Filename: utils
# To change this template, choose Tools | Templates
# and open the template in the editor.
__author__ = 'masuo'
__date__ = "2018/04/02"

# separate(folder)は訓練用、評価用に分ける関数
# movie_separate(folder,num)は当該ファイル（訓練用）、その他訓練用、当該ファイル（評価用）、その他評価用に分ける
# folder_create(folder)


def folder_create(folder):
    if not os.path.isdir(folder):
        os.mkdir(folder)
    return

def folder_delete(folder):
    if os.path.isdir(folder):
        shutil.rmtree(folder)

def ID_reading(dataset_folder,idx):
    df = pd.read_csv(os.path.join(dataset_folder, "dataset" + "_" + str(idx) + "." + "csv"),encoding="utf-8")
    # ファイル名一覧
    train_list = df["train"].dropna()
    test_list = df["test"].dropna()
    return train_list, test_list

def clopper_pearson(k, n, alpha):
    alpha2 = (1 - alpha) / 2
    lower = scipy.stats.beta.ppf(alpha2, k, n - k + 1)
    upper = scipy.stats.beta.ppf(1 - alpha2, k + 1, n - k)
    return (lower, upper)

def folder_clean(img_root):
    folder_list = os.listdir(img_root)
    for folder in folder_list:
        folderpath = os.path.join(img_root, folder)
        if folder == "desktop.ini":
            os.remove(folderpath)
        else:
            file_list = os.listdir(folderpath)
            for file in file_list:
                if file == "desktop.ini":
                    fpath = os.path.join(folderpath, file)
                    os.remove(fpath)
    return

def split_array(ar, n_group):
    for i_chunk in range(n_group):
        yield ar[i_chunk * len(ar) // n_group:(i_chunk + 1) * len(ar) // n_group]

def list_shuffle(a,b,seed):
    np.random.seed(seed)
    l = list(zip(a, b))
    np.random.shuffle(l)
    a1, b1 = zip(*l)
    a2 = list(a1)
    b2 = list(b1)
    return a2,b2

# root/00_normal/画像を見に行く。
# filepathのリストと
# tagのカテゴリー化済みのarrayが出力される。
def fpath_tag_making(root,classes):
    # train
    seed = 1
    folder_list = os.listdir(root)
    fpath_list = []
    tag_list = []
    # train/00_tgt
    for i, folder in enumerate(folder_list):
        folder_path = os.path.join(root, folder)
        file_list = os.listdir(folder_path)
        for file in file_list:
            fpath = os.path.join(folder_path, file)
            fpath_list.append(fpath)
            tag_list.append(i)
    fpath_list, tag_list = list_shuffle(fpath_list, tag_list,seed)
    tag_array = np.array(tag_list)
    tag_array = np_utils.to_categorical(tag_array, classes)
    return fpath_list,tag_array


def model_compile(model, loss, optimizer):
    model.compile(loss = loss, optimizer = optimizer, metrics = ["accuracy"])
    return

def model_load(model,model_folder,idx):
    model_files= glob.glob(os.path.join(model_folder, "weights_" + str(idx) + "_*"))
    model_fpath = model_files[-1]
    model.load_weights(model_fpath)
    return

def model_delete(model,model_folder,idx):
    model_files= glob.glob(os.path.join(model_folder, "weights_" + str(idx) + "_*"))
    for model_fpath in model_files[:-1]:
        os.remove(model_fpath)
    return

    # 訓練用データ数のカウント
def num_count(root):
    dic = {}
    folder_list = os.listdir(root)
    for folder in folder_list:
        folder_path = os.path.join(root, folder)
        file_list = os.listdir(folder_path)
        dic[folder] = len(file_list)
    return dic

def read_img(fpath,h,w):
    X = np.array(
        cv2.resize(cv2.imread(fpath), (h, w)) / 255.0,
        dtype=np.float32)
    return X
