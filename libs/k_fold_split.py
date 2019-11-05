#!/usr/bin/env python
# -*- coding: utf-8 -*-

# データセットの交差検証

import pandas as pd
import numpy as np
import dataclasses
from collections import defaultdict
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
        train_index : array-like, shape(分割数, ファイル数)
            学習用として分けられたi分割目のXのインデックス
        test_index : array-like, shape(分割数, ファイル数)
            テスト用として分けられたi分割目のXのインデックス
        """


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
        train_index : array-like, shape(分割数, ファイル数)
            学習用として分けられたi分割目のXのインデックス
        test_index : array-like, shape(分割数, ファイル数)
            テスト用として分けられたi分割目のXのインデックス
        """

        # 初期化
        ## グループがない場合はファイル名をグループ名とする
        ## ユニークなグループ名を取得
        if groups is None:
            groups = X
        unique_group_list = list(set(groups))

        ## ラベルの数と種類を取得
        labels_list = list(set(y))
        labels_num = len(labels_list)
        y_count = np.zeros(labels_num)
        for _y in y:
            y_count[labels_list.index(_y)] += 1

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
            np.random.shuffle(unique_group_list)

        # グループ層化K分割
        # 各分割群のラベル数を調べ，
        # ラベル数の標準偏差が最小になるようにデータを割り当てる
        for unique_group in tqdm(unique_group_list, desc='k-fold_split'):
            best_fold = None
            min_value = None
            for i in range(self.n_splits):
                group_y_count_fold[i] += group_y_count_dict[unique_group]
                std_per_label = []
                for label in range(labels_num):
                    label_std = np.std([group_y_count_fold[i][label]
                                        / y_count[label]
                                        for i in range(self.n_splits)])
                    std_per_label.append(label_std)
                group_y_count_fold[i] -= group_y_count_dict[unique_group]
                value = np.mean(std_per_label)
        
                if min_value is None or value < min_value:
                    min_value = value
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
            df_train = df.iloc[train_index]
            df_test = df.iloc[test_index]

            ## 分割されたデータの情報を出力
            df_train.to_csv(f'{self.split_info_folder}/train_{i}.csv',
                            index=False, encoding='utf-8')
            df_test.to_csv(f'{self.split_info_folder}/test_{i}.csv',
                           index=False, encoding='utf-8')
            
            df_train_list.append(df_train)
            df_test_list.append(df_test)

        return df_train_list, df_test_list


    def k_fold_regressor(self, df, bins_num=None):
        """
        回帰問題においてグループ層化K分割を行い，分割の内訳をcsvで保存する
        数値ラベルを数値を基準にグループ化し，分布が均等になるようにK分割する

        Parameters
        ----------
        df : DataFrame(pandas)
            学習に使用するデータの情報
        bins_num : int or None
            疑似ラベルの分割数，Noneの場合，分割数はデータ数の平方根となる

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
        y = df[self.label_column].values

        ## 数値の分布が均等になるように分割するために疑似ラベルを作成
        if bins_num is None:
            bins_num = int(len(X) ** 0.5) + 1
        bins = np.linspace(min(y), max(y), bins_num)
        y_pseudo = np.digitize(y, bins) - 1
        y_pseudo[np.argmax(y)] -= 1
        if self.group_column == 'None':
            groups = None
        else:
            groups = df[self.group_column].values
        df_train_list = []
        df_test_list = []
        for i, (train_index, test_index) in enumerate(self.split(X, y_pseudo, groups)):
            df_train = df.iloc[train_index]
            df_test = df.iloc[test_index]

            ## 分割されたデータの情報を出力
            df_train.to_csv(f'{self.split_info_folder}/train_{i}.csv',
                            index=False, encoding='utf-8')
            df_test.to_csv(f'{self.split_info_folder}/test_{i}.csv',
                           index=False, encoding='utf-8')
            
            df_train_list.append(df_train)
            df_test_list.append(df_test)

        return df_train_list, df_test_list