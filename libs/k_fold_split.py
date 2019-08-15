#!/usr/bin/env python
# -*- coding: utf-8 -*-

# データセットの交差検証

import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold
from .utils.folder import folder_create
from tqdm import tqdm


def stratified_k_fold(k, csv_config, split_info_folder, df, hasID):
    '''
    層化k分割を行い、分割情報をcsvに保存する
    image_classifier.pyで使うことを想定
    '''

    RANDOM_SEED = 1

    filename_column = csv_config["image_filename_column"]
    label_column = csv_config["label_column"]

    # ID列がない場合はファイル名列をID列として扱う
    if hasID:
        id_column = csv_config["ID_column"]
    else:
        id_column = filename_column

    # ID_uniqueは個人ID, y_uniqueは分類ラベルのリスト
    # 個人IDと分類ラベルは対応し、ID_uniqueは一意である
    # [！] 同じ個人IDは全て同じ分類ラベルを持っている仮定に基づく
    ID_unique_list = df[id_column].unique().tolist()
    y_unique_list = []
    for ID in tqdm(ID_unique_list, desc="finding same ID"):
        # 同じ個人IDを持つ行を抽出
        queried_df = df[df[id_column] == ID]
        # 最初に見つかった分類ラベルを使用する
        label = queried_df[label_column].values.astype(str).tolist()[0]
        y_unique_list.append(label)

    # 個人IDと分類ラベルの対応を用いてデータ分割を行う
    skf = StratifiedKFold(n_splits=k, shuffle=True,
                          random_state=RANDOM_SEED)
    filelist_for_train = []
    filelist_for_test = []
    for (train_index, test_index) in tqdm(skf.split(ID_unique_list,
                                                    y_unique_list),
                                          desc="splitting train/test"):
        IDs_for_train = []
        IDs_for_test = []
        for id_index in train_index:
            IDs_for_train.append(ID_unique_list[id_index])
        for id_index in test_index:
            IDs_for_test.append(ID_unique_list[id_index])

        # 分類ラベルをkey、ファイル名のリストをvalueとした辞書
        label_filelist4Train_dic = {label: [] for label in y_unique_list}
        label_filelist4Test_dic = {label: [] for label in y_unique_list}

        for ID in tqdm(IDs_for_train, desc="splitting train"):
            # 同じ個人IDを持つ行を抽出
            queried_df = df[df[id_column] == ID]
            filename_list = queried_df[filename_column].values.tolist()
            label_list = queried_df[label_column].values.astype(str).tolist()
            for filename, label in zip(filename_list, label_list):
                label_filelist4Train_dic[label].append(filename)

        for ID in tqdm(IDs_for_test, desc="splitting train"):
            # 同じ個人IDを持つ行を抽出
            queried_df = df[df[id_column] == ID]
            filename_list = queried_df[filename_column].values.tolist()
            label_list = queried_df[label_column].values.astype(str).tolist()
            for filename, label in zip(filename_list, label_list):
                label_filelist4Test_dic[label].append(filename)

        filelist_for_train.append(label_filelist4Train_dic)
        filelist_for_test.append(label_filelist4Test_dic)

    # 分割情報が書かれたcsvを保存するフォルダ
    folder_create(split_info_folder)
    unique_label_list = df[label_column].unique().astype(str).tolist()

    df_train_list = []
    df_test_list = []

    # スプリット順にcsvに出力
    for idx in range(k):
        df_train = pd.DataFrame()
        df_test = pd.DataFrame()
        # 行（分類ラベル）ごとに列(ファイル名)を追加していく
        for label in unique_label_list:
            train_name = filelist_for_train[idx][label]
            ds_train = pd.Series(train_name)
            df_train = pd.concat([df_train,
                                  pd.DataFrame(ds_train, columns=[label])],
                                 axis=1)

            test_name = filelist_for_test[idx][label]
            ds_test = pd.Series(test_name)
            df_test = pd.concat([df_test,
                                 pd.DataFrame(ds_test, columns=[label])],
                                axis=1)

        df_train.to_csv(f"{split_info_folder}/train_{idx}.csv",
                        index=False, encoding="utf-8")
        df_test.to_csv(f"{split_info_folder}/test_{idx}.csv",
                       index=False, encoding="utf-8")

        df_train_list.append(df_train)
        df_test_list.append(df_test)

    return df_train_list, df_test_list


def simple_k_fold(k, csv_config, split_info_folder, df, hasID):
    '''
    k分割交差検証を行い、分割情報をcsvに保存する
    image_regressor.pyで使うことを想定
    '''

    RANDOM_SEED = 1
    filename_column = csv_config["image_filename_column"]
    label_column = csv_config["label_column"]

    # ID列がない場合はファイル名列をID列として扱う
    if hasID:
        id_column = csv_config["ID_column"]
    else:
        id_column = filename_column

    # ID_uniqueは一意な個人IDのリスト
    ID_unique_list = df[id_column].unique().tolist()
    for ID in tqdm(ID_unique_list, desc="finding same ID"):
        # 同じ個人IDを持つ行を抽出
        queried_df = df[df[id_column] == ID]

    # 個人IDを用いてデータ分割を行う
    kf = KFold(n_splits=k, shuffle=True,
               random_state=RANDOM_SEED)
    filelist_for_train = []
    filelist_for_test = []
    tergetlist_for_train = []
    tergetlist_for_test = []
    for (train_index, test_index) in tqdm(kf.split(ID_unique_list),
                                          desc="splitting train/test"):
        IDs_for_train = []
        IDs_for_test = []
        splited_filelist_for_train = []
        splited_filelist_for_test = []
        splited_tergetlist_for_train = []
        splited_tergetlist_for_test = []

        for id_index in train_index:
            IDs_for_train.append(ID_unique_list[id_index])
        for id_index in test_index:
            IDs_for_test.append(ID_unique_list[id_index])

        for ID in tqdm(IDs_for_train, desc="splitting train"):
            # 同じ個人IDを持つ行を抽出
            queried_df = df[df[id_column] == ID]
            filename_list = queried_df[filename_column].values.tolist()
            label_list = queried_df[label_column].values.astype(str).tolist()
            for filename, label in zip(filename_list, label_list):
                splited_filelist_for_train.append(filename)
                splited_tergetlist_for_train.append(label)

        for ID in tqdm(IDs_for_test, desc="splitting test"):
            # 同じ個人IDを持つ行を抽出
            queried_df = df[df[id_column] == ID]
            filename_list = queried_df[filename_column].values.tolist()
            label_list = queried_df[label_column].values.astype(str).tolist()
            for filename, label in zip(filename_list, label_list):
                splited_filelist_for_test.append(filename)
                splited_tergetlist_for_test.append(label)

        filelist_for_train.append(splited_filelist_for_train)
        filelist_for_test.append(splited_filelist_for_test)
        tergetlist_for_train.append(splited_tergetlist_for_train)
        tergetlist_for_test.append(splited_tergetlist_for_test)

    # 分割情報が書かれたcsvを保存するフォルダ
    folder_create(split_info_folder)

    df_train_list = []
    df_test_list = []

    # スプリット順にcsvに出力
    for idx in range(k):
        df_train = pd.DataFrame()
        df_test = pd.DataFrame()

        # 列(ファイル名)を追加していく
        train_name = filelist_for_train[idx]
        terget_val = tergetlist_for_train[idx]
        ds_train = pd.Series(train_name)
        ds_terget = pd.Series(terget_val)
        df_train = pd.concat([df_train,
                              pd.DataFrame(ds_train, columns=["filename"]),
                              pd.DataFrame(ds_terget, columns=["terget"])],
                             axis=1)

        test_name = filelist_for_test[idx]
        terget_val = tergetlist_for_test[idx]
        ds_test = pd.Series(test_name)
        ds_terget = pd.Series(terget_val)
        df_test = pd.concat([df_test,
                             pd.DataFrame(ds_test, columns=["filename"]),
                             pd.DataFrame(ds_terget, columns=["terget"])],
                            axis=1)

        df_train.to_csv(f"{split_info_folder}/train_{idx}.csv",
                        index=False, encoding="utf-8")
        df_test.to_csv(f"{split_info_folder}/test_{idx}.csv",
                       index=False, encoding="utf-8")

        df_train_list.append(df_train)
        df_test_list.append(df_test)

    return df_train_list, df_test_list
