#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 層化k分割の交差検証

import os
# import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from utils import folder_create
# from utils import split_array
# import random


class Split:
    def __init__(self, k, source_folder, dataset_folder):
        '''
        コンストラクタ、分割数、元フォルダ、出力先のデータセットフォルダを
        '''

        self.k = k
        self.source_folder = source_folder
        self.dataset_folder = dataset_folder
        self.random_seed = 1

    def k_fold_split(self):
        '''
        画像用フォルダ
        '''
        folder_create(self.dataset_folder)
        # 00_normal,01_Glaなど
        folder_list = os.listdir(self.source_folder)
        nb_classes = len(folder_list)
        X = []
        y = []
        for i, folder in enumerate(folder_list):
            # img/00_normal
            folder_path = os.path.join(self.source_folder, folder)
            file_list = os.listdir(folder_path)
            tag_list = [i for x in range(len(file_list))]
            X.extend(file_list)
            y.extend(tag_list)

        X = np.array(X)
        y = np.array(y)

        # shuffleする、リストにスプリット順に、保存されていく。
        skf = StratifiedKFold(n_splits=self.k, shuffle=True,
                              random_state=self.random_seed)
        train_list = []
        test_list = []
        for (i, (train_index, test_index)) in enumerate(skf.split(X, y)):
            train_folder_list = [[] for i in range(nb_classes)]
            test_folder_list = [[] for i in range(nb_classes)]
            for index in train_index:
                filename = X[index]
                tag = y[index]
                train_folder_list[tag].append(filename)
            for index in test_index:
                filename = X[index]
                tag = y[index]
                test_folder_list[tag].append(filename)
            train_list.append(train_folder_list)
            test_list.append(test_folder_list)

        # スプリット順にcsvに出力
        # trainとtestは別ファイル
        # 00_normalみたいに出力されていく。
        for idx in range(self.k):
            df_train = pd.DataFrame()
            df_test = pd.DataFrame()

            # 行（フォルダ）ごとに列を追加していく。
            for col_idx in range(nb_classes):
                folder_name = folder_list[col_idx]
                train_name = train_list[idx][col_idx]
                ds_train = pd.Series(train_name)
                test_name = test_list[idx][col_idx]
                ds_test = pd.Series(test_name)
                df_train = pd.concat([df_train, pd.DataFrame(
                    ds_train, columns=[folder_name])], axis=1)
                df_test = pd.concat([df_test, pd.DataFrame(
                    ds_test, columns=[folder_name])], axis=1)

            df_train.to_csv(self.dataset_folder + "/" + "train" + "_" + str(idx) + ".csv",
                            index=False, encoding="utf-8")
            df_test.to_csv(self.dataset_folder + "/" + "test" + "_" + str(idx) + ".csv",
                           index=False, encoding="utf-8")
        return

    def k_fold_split_unique(self):
        '''
        画像用フォルダ IDを考慮した分割
        '''
        folder_create(self.dataset_folder)
        # 00_normal,01_Glaなど
        folder_list = os.listdir(self.source_folder)
        nb_classes = len(folder_list)
        X = []
        y = []
        for i, folder in enumerate(folder_list):
            # img/00_normal
            folder_path = os.path.join(self.source_folder, folder)
            file_list = os.listdir(folder_path)
            tag_list = [i for x in range(len(file_list))]
            X.extend(file_list)
            y.extend(tag_list)

        # ファイル名が "患者ID_画像ID" になっていると仮定
        # 患者IDのみ抽出
        X_unique = []
        y_unique = []
        for (i, j) in zip(X, y):
            if((i[:-6] in X_unique) == False):
                X_unique.append(i[:-6])
                y_unique.append(j)

        X_unique = np.array(X_unique)
        y_unique = np.array(y_unique)
        # shuffleする、リストにスプリット順に、保存されていく。
        skf = StratifiedKFold(n_splits=self.k, shuffle=True,
                              random_state=self.random_seed)
        train_list = []
        test_list = []
        # 患者IDのみでデータ分割
        for (i, (train_index, test_index)) in enumerate(skf.split(X_unique, y_unique)):
            train_folder_list = [[] for i in range(nb_classes)]
            test_folder_list = [[] for i in range(nb_classes)]
            for index in train_index:
                # 患者IDが同じデータを探し，リストに入れる(非効率なので要修正)
                for (filename, tag) in zip(X, y):
                    if(filename[:-6] == X_unique[index]):
                        train_folder_list[tag].append(filename)
            for index in test_index:
                # 患者IDが同じデータを探し，リストに入れる(非効率なので要修正)
                for (filename, tag) in zip(X, y):
                    if(filename[:-6] == X_unique[index]):
                        test_folder_list[tag].append(filename)
            train_list.append(train_folder_list)
            test_list.append(test_folder_list)

        # スプリット順にcsvに出力
        # trainとtestは別ファイル
        # 00_normalみたいに出力されていく。
        for idx in range(self.k):
            df_train = pd.DataFrame()
            df_test = pd.DataFrame()
            # 行（フォルダ）ごとに列を追加していく。
            for col_idx in range(nb_classes):
                folder_name = folder_list[col_idx]
                train_name = train_list[idx][col_idx]
                ds_train = pd.Series(train_name)
                test_name = test_list[idx][col_idx]
                ds_test = pd.Series(test_name)
                df_train = pd.concat([df_train, pd.DataFrame(
                    ds_train, columns=[folder_name])], axis=1)
                df_test = pd.concat([df_test, pd.DataFrame(
                    ds_test, columns=[folder_name])], axis=1)

            df_train.to_csv(self.dataset_folder + "/" + "train" + "_" + str(idx) + ".csv",
                            index=False, encoding="utf-8")
            df_test.to_csv(self.dataset_folder + "/" + "test" + "_" + str(idx) + ".csv",
                           index=False, encoding="utf-8")
        return
