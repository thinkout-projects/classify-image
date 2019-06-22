#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 層化k分割の交差検証

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from .utils.folder import folder_create


class Split:
    def __init__(self, k, csv_config, split_info_folder, df, classes):
        '''
        コンストラクタ、引数からメンバを作成
        '''

        self.k = k
        self.filename_column = csv_config["image_filename_column"]
        self.label_column = csv_config["label_column"]
        self.id_column = csv_config["ID_column"]
        self.split_info_folder = split_info_folder
        self.random_seed = 1
        self.df = df.sort_values(self.id_column)  # この時点でid順にソートして扱う
        self.classes = classes

    def k_fold_split(self):
        '''
        実際にk_fold_splitを行い、分割情報をcsvに保存する
        '''
        # 分割情報が書かれたcsvを保存するフォルダ
        folder_create(self.split_info_folder)

        # Xはfilename, yは分類ラベルのリスト
        X = self.df[self.filename_column].values.tolist()
        y = self.df[self.label_column].values.tolist()
        unique_label_list = self.df[self.label_column].unique().tolist()

        # ID_uniqueは個人ID, y_uniqueは分類ラベルのリスト
        # 個人IDと分類ラベルは対応し、ID_uniqueは一意である
        # [！] 同じ個人IDは全て同じ分類ラベルを持っている仮定に基づく
        ID_unique_list = self.df[self.id_column].unique().tolist()
        y_unique_list = []
        for ID in ID_unique_list:
            # 同じ個人IDを持つ行を抽出
            queried_df = self.df[self.df[self.id_column] == ID]
            # 最初に見つかった分類ラベルを使用する
            label = queried_df[self.label_column].values.tolist()[0]
            y_unique_list.append(label)

        # 個人IDと分類ラベルの対応を用いてデータ分割を行う
        skf = StratifiedKFold(n_splits=self.k, shuffle=True,
                              random_state=self.random_seed)
        filelist_for_train = []
        filelist_for_test = []
        for (i, (ID_for_train, ID_for_test)) in enumerate(skf.split(ID_unique_list,
                                                                    y_unique_list)):
            # 分類ラベルをkey、ファイル名のリストをvalueとした辞書
            label_filelist4Train_dic = {label: [] for label in y_unique_list}
            label_filelist4Test_dic = {label: [] for label in y_unique_list}

            for ID in ID_for_train:
                # 同じ個人IDを持つ行を抽出
                queried_df = self.df[self.df[self.id_column] == ID]
                filename_list = queried_df[self.filename_column].values.tolist()
                label_list = queried_df[self.label_column].values.tolist()
                for filename, label in zip(filename_list, label_list):
                    label_filelist4Train_dic[label].append(filename)

            for ID in ID_for_test:
                # 同じ個人IDを持つ行を抽出
                queried_df = self.df[self.df[self.id_column] == ID]
                filename_list = queried_df[self.filename_column].values.tolist()
                label_list = queried_df[self.label_column].values.tolist()
                for filename, label in zip(filename_list, label_list):
                    label_filelist4Test_dic[label].append(filename)

            filelist_for_train.append(label_filelist4Train_dic)
            filelist_for_test.append(label_filelist4Test_dic)

        # スプリット順にcsvに出力
        for idx in range(self.k):
            df_train = pd.DataFrame()
            df_test = pd.DataFrame()
            # 行（分類ラベル）ごとに列(ファイル名)を追加していく
            for label in unique_label_list:
                train_name = filelist_for_train[idx][label]
                ds_train = pd.Series(train_name)
                df_train = pd.concat([df_train, pd.DataFrame(
                    ds_train, columns=[label])], axis=1)

                test_name = filelist_for_test[idx][label]
                ds_test = pd.Series(test_name)
                df_test = pd.concat([df_test, pd.DataFrame(
                    ds_test, columns=[label])], axis=1)

            df_train.to_csv(self.split_info_folder + "/"
                            + "train" + "_" + str(idx) + ".csv",
                            index=False, encoding="utf-8")
            df_test.to_csv(self.split_info_folder + "/"
                           + "test" + "_" + str(idx) + ".csv",
                           index=False, encoding="utf-8")

        return

    def k_fold_split_simple(self):
        self.id_column = self.filename_column
        self.k_fold_split()
        return
