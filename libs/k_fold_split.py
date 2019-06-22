#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 層化k分割の交差検証

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from .utils.folder import folder_create
from .utils.utils import printWithDate
from .data_generator import define_data_aug


class Split:
    def __init__(self, k, csv_config, split_info_folder, df, classes):
        '''
        コンストラクタ、分割数、元フォルダ、出力先のデータセットフォルダを
        '''

        self.k = k
        self.simple = csv_config['csv_simple_split']
        self.filename_column = csv_config["csv_column_filename"]
        self.label_column = csv_config["csv_column_label"]
        self.id_column = csv_config["csv_column_ID"]
        self.split_info_folder = split_info_folder
        self.random_seed = 1
        self.df = df.sort_values(self.id_column)  # この時点でid順にソートして扱う
        self.classes = classes

    def k_fold_split(self):
        '''
        分割情報をcsvで保存する用フォルダ simpleがTrueならIDを考慮して分割
        '''
        folder_create(self.split_info_folder)

        # Xはfilename, yはgradeのリスト
        X = self.df[self.filename_column].values.tolist()
        y = self.df[self.label_column].values.tolist()
        id_from_df = self.df[self.id_column].values.tolist()

        X_unique = []
        y_unique = []
        printWithDate("spliting - loop 1/3")
        for i,candid in enumerate(id_from_df):
            if((candid in X_unique) is False):
                X_unique.append(candid)
                y_unique.append(y[i])

        # shuffleする、リストにスプリット順に、保存されていく。
        skf = StratifiedKFold(n_splits=self.k, shuffle=True,
                              random_state=self.random_seed)
        train_list = []
        test_list = []
        # 患者IDのみでデータ分割
        printWithDate("spliting - loop 2/3")
        for (i, (train_index, test_index)) in enumerate(skf.split(X_unique,
                                                                  y_unique)):
            train_folder_list = [[] for i in range(self.classes)]
            test_folder_list = [[] for i in range(self.classes)]
            for index in train_index:
                if self.simple is False:
                    # 患者IDが同じデータを探し，リストに入れる(非効率なので要修正)
                    for (filename, tag) in zip(X, y):
                        if(filename == X_unique[index]):
                            train_folder_list[tag].append(filename)
                else:
                    filename = X[index]
                    tag = y[index]
                    train_folder_list[tag].append(filename)
            for index in test_index:
                if self.simple is True:
                    # 患者IDが同じデータを探し，リストに入れる(非効率なので要修正)
                    for (filename, tag) in zip(X, y):
                        if(filename == X_unique[index]):
                            test_folder_list[tag].append(filename)
                else:
                    filename = X[index]
                    tag = y[index]
                    test_folder_list[tag].append(filename)
            train_list.append(train_folder_list)
            test_list.append(test_folder_list)

        # スプリット順にcsvに出力
        # trainとtestは別ファイル
        # 00_normalみたいに出力されていく。
        printWithDate("spliting - loop 3/3")
        for idx in range(self.k):
            df_train = pd.DataFrame()
            df_test = pd.DataFrame()
            # 行（フォルダ）ごとに列を追加していく。
            for col_idx in range(self.classes):
                folder_name = col_idx
                train_name = train_list[idx][col_idx]
                ds_train = pd.Series(train_name)
                test_name = test_list[idx][col_idx]
                ds_test = pd.Series(test_name)
                df_train = pd.concat([df_train, pd.DataFrame(
                    ds_train, columns=[folder_name])], axis=1)
                df_test = pd.concat([df_test, pd.DataFrame(
                    ds_test, columns=[folder_name])], axis=1)

            df_train.to_csv(self.split_info_folder + "/"
                            + "train" + "_" + str(idx) + ".csv",
                            index=False, encoding="utf-8")
            df_test.to_csv(self.split_info_folder + "/"
                           + "test" + "_" + str(idx) + ".csv",
                           index=False, encoding="utf-8")

        return df_train, df_test
