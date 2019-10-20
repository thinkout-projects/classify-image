#!/usr/bin/env python
# -*- coding: utf-8 -*-

# データセットの交差検証

import pandas as pd
import numpy as np
import dataclasses
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold, KFold
from .utils.folder import folder_create
from tqdm import tqdm



@dataclasses.dataclass
class Stratified_group_k_fold:
    """
    データをグループ層化K分割するときのパラメータを保持する
    """
    csv_config: dict # 学習に使用するデータの情報が書かれたcsvの情報
    split_info_folder : str # 分割されたファイルの内訳を保存するフォルダ名
    n_splits: int = 5 # 分割数
    shuffle: bool = False # シャッフルするかどうか
    random_state: int = None # ランダムシード


    def __post_init__(self):
        self.filename_column = self.csv_config["image_filename_column"] # ファイル列
        self.label_column = self.csv_config["label_column"] # ラベル列
        self.group_column = self.csv_config["ID_column"] # グループ列


    def split(self, X, y, groups=None):
        """
        グループ層化K分割する

        Parameters
        ----------
        X : array-like, shape(ファイル数,)
            分割するファイル名
        y : array-like, shape(ファイル数,)
            分割するファイル名のラベル
        groups : None or array-like, shape(ファイル数,)
            分割するファイルのグループ名
            Noneの場合はただの層化K分割となる

        Yields
        -------
        train : array-like, shape(分割数, ファイル数)
            学習用として分けられたi分割目のXのインデックス
        test : array-like, shape(分割数, ファイル数)
            テスト用として分けられたi分割目のXのインデックス

        """

        # 初期化
        ## グループがない場合はファイル名をグループ名とする
        if groups is None:
            groups = X

        ## ラベルの数と種類を取得
        labels_list = list(set(y))
        labels_num = len(labels_list)
        y_count = np.zeros(labels_num)
        for _y in y:
            y_count += labels_list.index(_y)

        ## グループとファイル名の対応辞書，ファイル名とラベルの対応辞書，
        ## グループとラベルの数および種類の対応辞書を作成
        group_X_dict = defaultdict(list)
        X_y_dict = defaultdict(list)
        group_y_count_dict = defaultdict(lambda: np.zeros(labels_num))

        for _X, _y, _groups in zip(X, y, groups):
            group_X_dict[_groups].append(_X)
            idx = labels_list.index(_y)
            X_y_dict[_X] = idx
            group_y_count_dict[_groups][idx] += 1
            
        ## 分割後の情報を保存する変数の初期化
        group_X_fold = [[] for i in range(self.n_splits)]
        group_y_count_fold = [np.zeros(labels_num)
                              for i in range(self.n_splits)]

        # グループを1単位としてシャッフル
        if self.shuffle is True:
            np.random.seed(seed=self.random_state)
            unique_group_list = list(set(groups))
            np.random.shuffle(unique_group_list)

        # グループ層化K分割
        # 各分割群のラベル数を調べ，
        # ラベル数の標準偏差が最小になるようにデータを割り当てる
        for unique_group in tqdm(unique_group_list, desc='k-fold_split'):
            best_fold = None
            min_eval = None
            for i in range(self.n_splits):
                group_y_count_fold[i] += group_y_count_dict[unique_group]
                std_per_label = []
                for label in range(labels_num):
                    label_std = np.std([group_y_count_fold[i][label]
                                        / y_count[label]
                                        for i in range(self.n_splits)])
                    std_per_label.append(label_std)
                group_y_count_fold[i] -= group_y_count_dict[unique_group]
                eval = np.mean(std_per_label)
        
                if min_eval is None or eval < min_eval:
                    min_eval = eval
                    best_fold = i

            group_y_count_fold[best_fold] += group_y_count_dict[unique_group]
            group_X_fold[best_fold] += group_X_dict[unique_group]

        # i番目の分割群をテストデータ，残りを学習データとする
        X_set = set(X)
        for i in range(self.n_splits):
            X_train = X_set - set(group_X_fold[i])
            X_test = set(group_X_fold[i])

            train_index = [i for i, _X in enumerate(X) if _X in X_train]
            test_index = [i for i, _X in enumerate(X) if _X in X_test]

            yield train_index, test_index

        
    def k_fold_classifier(self, df):
        """
        分類問題においてグループ層化K分割を行い，分割の内訳をcsvで保存する

        Parameters
        ----------
        df : DataFrame(pandas)
            学習に使用するデータの情報

        Returns
        -------
        df_train_list : array-like[DataFrame(pandas)], shape(分割数,)
            学習用として分けられたデータ
        df_test_list : array-like, shape(分割数, ファイル数)
            テスト用として分けられたデータ
        """

        # グループ層化K分割
        folder_create(self.split_info_folder)
        X = df[self.filename_column].values
        y = list(map(str, df[self.label_column].values))
        if self.group_column == 'None':
            groups = None
        else:
            groups = df[self.group_column].values
        df_train_list = []
        df_test_list = []
        for i, (train_index, test_index) in enumerate(self.split(X, y, groups)):
            train_dict = defaultdict(list)
            test_dict = defaultdict(list)
            for idx in train_index:
                train_dict[y[idx]].append(X[idx])
            for idx in test_index:
                test_dict[y[idx]].append(X[idx])

            df_train = df_train = pd.DataFrame()
            for key in train_dict.keys():
                ds_train = pd.Series(train_dict[key])
                df_train = pd.concat([df_train,
                                      pd.DataFrame(ds_train, columns=[key])],
                                     axis=1)
            df_test = df_test = pd.DataFrame()
            for key in test_dict.keys():
                ds_test = pd.Series(test_dict[key])
                df_test = pd.concat([df_test,
                                     pd.DataFrame(ds_test, columns=[key])],
                                    axis=1)

            ## 分割されたデータの情報を出力
            df_train.to_csv(f"{self.split_info_folder}/train_{i}.csv",
                            index=False, encoding="utf-8")
            df_test.to_csv(f"{self.split_info_folder}/test_{i}.csv",
                           index=False, encoding="utf-8")
            
            df_train_list.append(df_train)
            df_test_list.append(df_test)

        return df_train_list, df_test_list


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
