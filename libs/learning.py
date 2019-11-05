#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import dataclasses
import numpy as np
from .utils.folder import folder_create

# plot用に
import matplotlib.pyplot as plt

# main関数
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm

from .data_generator import Trans
from .utils.utils import read_img


# モニターのないサーバーでも動作するように
plt.switch_backend('agg')


@dataclasses.dataclass
class Learning:
    """
    ニューラルネットワークモデルの学習を行うときのパラメータなどを保持する
    """

    directory: str # 画像が保存されているフォルダ
    csv_config: dict # 学習に使用するデータの情報が書かれたcsvの情報
    df_train: None # 学習データの情報がかかれたDataFrame
    df_validation: None # 検証データの情報がかかれたDataFrame
    label_list: list # ラベル名
    idx: int # k-fold splitの何番目か
    base_augmentation: dict # ライブラリを使用した画像増幅のパラメータ
    extra_augmentation: dict # 自作の画像増幅のパラメータ
    image_size: tuple # 入力画像サイズ
    classes: int # 分類クラス数
    batch_size: int # バッチサイズ
    model_folder: str # モデルの重みを保存するフォルダ
    epochs: int # エポック数

    
    def __post_init__(self):
        self.filename_column = self.csv_config["image_filename_column"] # ファイル列
        self.label_column = self.csv_config["label_column"] # ラベル列
        self.group_column = self.csv_config["ID_column"] # グループ列
        if self.classes == 1:
            self.class_mode = 'other'
        else:
            self.class_mode = 'categorical'
        for key in self.base_augmentation:
            if (self.base_augmentation[key] != 'True'
                and self.base_augmentation[key] != 'False'):
                self.base_augmentation[key] = float(self.base_augmentation[key])
            else:
                self.base_augmentation[key] = bool(self.base_augmentation[key])


    def balance_making(self):
        balance_list = []
        num = 0
        for i in range(self.classes):
            cnt = 0
            des = [1 if idx == i else 0 for idx in range(self.classes)]
            for idx in range(len(self.y_val)):
                if (list(self.y_val[idx])) == des:
                    cnt += 1
                    num += 1
            balance_list.append(cnt)
        dic = {}
        for i in range(self.classes):
            ans = num / (self.classes * balance_list[i])
            dic[i] = ans
        print(dic)
        return dic


    def train(self, model):
        # modelフォルダを作成
        folder_create(self.model_folder)
        model_file = "weights_" + str(self.idx) + "_epoch{epoch:02}.hdf5"

        # val_lossが最小になったときのみmodelを保存
        mc_cb = ModelCheckpoint(os.path.join(self.model_folder, model_file),
                                monitor='val_loss', verbose=1,
                                save_best_only=True, mode='min')

        # 学習が停滞したとき、学習率を0.2倍に
        rl_cb = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=3,
                                  verbose=1, mode='auto',
                                  min_delta=0.0001, cooldown=0, min_lr=0)

        # 学習が進まなくなったら、強制的に学習終了
        es_cb = EarlyStopping(monitor='loss', min_delta=0,
                              patience=5, verbose=1, mode='auto')

        # balance = Learning.balance_making(self)

        # ジェネレータの生成
        ## 学習データのジェネレータ
        augmentor = Trans(**self.extra_augmentation)
        datagen = ImageDataGenerator(
            rescale=1./255,
            preprocessing_function=augmentor.augment(),
            **self.base_augmentation)
        train_generator = datagen.flow_from_dataframe(
            dataframe=self.df_train, directory=self.directory,
            x_col=self.filename_column, y_col=self.label_column,
            target_size=self.image_size, class_mode=self.class_mode, 
            classes=self.label_list,
            batch_size=self.batch_size)
        step_size_train = train_generator.n // train_generator.batch_size
        ## 検証データのジェネレータ
        augmentor = Trans()
        datagen = ImageDataGenerator(
            rescale=1./255,
            preprocessing_function=augmentor.augment())
        validation_generator = datagen.flow_from_dataframe(
            dataframe=self.df_validation, directory=self.directory,
            x_col=self.filename_column, y_col=self.label_column,
            target_size=self.image_size, class_mode=self.class_mode, 
            classes=self.label_list,
            batch_size=self.batch_size)
        step_size_validation = \
            validation_generator.n // validation_generator.batch_size

        # 実際に学習⇒historyを作成
        history = model.fit_generator(
            train_generator, steps_per_epoch=step_size_train,
            epochs=self.epochs, verbose=1, callbacks=[mc_cb, rl_cb, es_cb],
            validation_data=validation_generator,
            validation_steps=step_size_validation,
            # class_weight = balance,
            workers=3)


        return history


    def predict(self, model):
        y_pred = []
        W_val = self.df_validation[self.filename_column].values
        if self.classes != 1:
            y_val = list(map(lambda x: self.label_list.index(x),
                             self.df_validation[self.label_column].values))
        else:
            y_val = self.df_validation[self.label_column].values.astype(float)
        for file in tqdm(W_val, desc='predict_validation'):
            image = read_img(os.path.join(self.directory, file),
                             self.image_size[0], self.image_size[1])
            image = np.expand_dims(image, axis=0)
            if self.classes != 1:
                y_pred.append(model.predict(image)[0])
            else:
                y_pred.append(model.predict(image)[0][0])

        return W_val, y_val, y_pred


def plot_hist(history, history_folder, metrics, idx):
    '''
    historyオブジェクト（model.fit_generatorから生まれる）の属性として
    .history["acc"]や.history["val_acc"]がある。
    元々のmovie名からタイトルを決定する
    '''

    folder_create(history_folder)
    history_file = "history" + "_" + str(idx) + "." + "jpg"
    fig, (axL, axR) = plt.subplots(ncols=2, figsize=(10, 4))
    L_title = metrics[0].upper() + metrics[1:] + '_vs_Epoch'
    axL.plot(history.history[metrics])
    axL.plot(history.history['val_'+metrics])

    # grid表示（格子を表示する）
    axL.grid(True)
    axL.set_title(L_title)
    axL.set_ylabel(metrics)
    axL.set_xlabel('epoch')

    # 凡例をtrainとtestとする。(plotした順番に並べる)
    axL.legend(['train', 'test'], loc='upper left')

    # summarize history for loss
    R_title = "Loss_vs_Epoch"
    axR.plot(history.history['loss'])
    axR.plot(history.history['val_loss'])
    axR.grid(True)
    axR.set_title(R_title)
    axR.set_ylabel('loss')
    axR.set_xlabel('epoch')
    axR.legend(['train', 'test'], loc='upper left')

    fig.savefig(os.path.join(history_folder, history_file))
    plt.clf()
    return
