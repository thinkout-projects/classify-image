#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 層化k分割の交差検証

import pandas as pd
from sklearn.model_selection import StratifiedKFold
from .utils.folder import folder_create


def k_fold_split(k, csv_config, split_info_folder, df, classes, hasID):
    '''
    k_fold_splitを行い、分割情報をcsvに保存する
    '''

    random_seed = 1

    filename_column = csv_config["image_filename_column"]
    label_column = csv_config["label_column"]

    # ID列がない場合はファイル名列をID列として扱う
    if hasID:
        id_column = filename_column
    else:
        id_column = csv_config["ID_column"]

    # ID_uniqueは個人ID, y_uniqueは分類ラベルのリスト
    # 個人IDと分類ラベルは対応し、ID_uniqueは一意である
    # [！] 同じ個人IDは全て同じ分類ラベルを持っている仮定に基づく
    ID_unique_list = df[id_column].unique().tolist()
    y_unique_list = []
    for ID in ID_unique_list:
        # 同じ個人IDを持つ行を抽出
        queried_df = df[df[id_column] == ID]
        # 最初に見つかった分類ラベルを使用する
        label = queried_df[label_column].values.tolist()[0]
        y_unique_list.append(label)

    # 個人IDと分類ラベルの対応を用いてデータ分割を行う
    skf = StratifiedKFold(n_splits=k, shuffle=True,
                          random_state=random_seed)
    filelist_for_train = []
    filelist_for_test = []
    for (i, (ID_for_train, ID_for_test)) in enumerate(skf.split(ID_unique_list,
                                                                y_unique_list)):
        # 分類ラベルをkey、ファイル名のリストをvalueとした辞書
        label_filelist4Train_dic = {label: [] for label in y_unique_list}
        label_filelist4Test_dic = {label: [] for label in y_unique_list}

        for ID in ID_for_train:
            # 同じ個人IDを持つ行を抽出
            queried_df = df[df[id_column] == ID]
            filename_list = queried_df[filename_column].values.tolist()
            label_list = queried_df[label_column].values.tolist()
            for filename, label in zip(filename_list, label_list):
                label_filelist4Train_dic[label].append(filename)

        for ID in ID_for_test:
            # 同じ個人IDを持つ行を抽出
            queried_df = df[df[id_column] == ID]
            filename_list = queried_df[filename_column].values.tolist()
            label_list = queried_df[label_column].values.tolist()
            for filename, label in zip(filename_list, label_list):
                label_filelist4Test_dic[label].append(filename)

        filelist_for_train.append(label_filelist4Train_dic)
        filelist_for_test.append(label_filelist4Test_dic)

    # 分割情報が書かれたcsvを保存するフォルダ
    folder_create(split_info_folder)
    unique_label_list = df[label_column].unique().tolist()

    # スプリット順にcsvに出力
    for idx in range(k):
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

        df_train.to_csv(f"{split_info_folder}/train_{str(idx)}.csv",
                        index=False, encoding="utf-8")
        df_test.to_csv(f"{split_info_folder}/test_{str(idx)}.csv",
                       index=False, encoding="utf-8")
    return
