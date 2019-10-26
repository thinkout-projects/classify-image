#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from .utils.folder import folder_create

# plot用に
import matplotlib.pyplot as plt

# main関数
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.callbacks import EarlyStopping
from .data_generator import Training
from .utils.utils import fpath_tag_making


# モニターのないサーバーでも動作するように
plt.switch_backend('agg')


class Learning(Training):
    '''
    Trainingのクラスをスーパークラスとして、サブクラスである学習クラスを作成
    '''

    def __init__(self, folder_names, idx,
                 pic_mode, train_num_mode_dic, size, classes, positive_label,
                 args_of_IDG, BATCH_SIZE, model_folder, model,
                 X_val, y_val, epochs, df_train):
        super().__init__(folder_names, idx,
                         pic_mode, train_num_mode_dic, size, classes,
                         positive_label, args_of_IDG, BATCH_SIZE, df_train)
        self.model_folder = model_folder
        self.model = model
        self.pic_mode = pic_mode
        self.X_val = X_val
        self.y_val = y_val
        self.epochs = epochs
        self.batch_size = BATCH_SIZE

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

    def learning_model(self):
        # modelフォルダを作成
        folder_create(self.model_folder)
        model_file = "weights_" + str(self.idx) + "_epoch{epoch:02}.hdf5"

        # val_lossが最小になったときのみmodelを保存
        mc_cb = ModelCheckpoint(os.path.join(self.model_folder, model_file),
                                monitor='val_loss', verbose=1,
                                save_best_only=True, mode='min')

        # 学習が停滞したとき、学習率を0.2倍に
        rl_cb = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5,
                                  verbose=1, mode='auto',
                                  epsilon=0.0001, cooldown=0, min_lr=0)

        # 学習が進まなくなったら、強制的に学習終了
        es_cb = EarlyStopping(monitor='loss', min_delta=0,
                              patience=20, verbose=1, mode='auto')

        # fpath_list,tag_listを作成する
        fpath_list, tag_array = fpath_tag_making(self.train_root, self.classes,
                                                 self.positive_label)
        # balance = Learning.balance_making(self)

        # 実際に学習⇒historyを作成
        history = self.model.fit_generator(Learning.datagen(
                                           self, fpath_list=fpath_list,
                                           tag_array=tag_array),
                                           int(len(fpath_list) /
                                               self.batch_size),
                                           epochs=self.epochs,
                                           # class_weight = balance,
                                           validation_data=(
            self.X_val, self.y_val),
            callbacks=[mc_cb, rl_cb, es_cb],
            workers=1,
            verbose=1)
        return history


def plot_hist(history, history_folder, metrics, idx):
    '''
    KerasのHistoryオブジェクトを受け取り、
    epochの進行に対するmetricsとlossの変化を記録したグラフを保存する。

    グラフは`history_folder`ディレクトリにhistory_`idx`.jpgの名前で保存される。
    '''

    folder_create(history_folder)
    history_file = f"history_{idx}.jpg"
    fig, (axL, axR) = plt.subplots(ncols=2, figsize=(10, 4))

    # [左側] metricsについてのグラフ
    L_title = metrics[0].upper() + metrics[1:] + '_vs_Epoch'
    axL.plot(history.history[metrics])
    axL.plot(history.history['val_'+metrics])
    axL.grid(True)
    axL.set_title(L_title)
    axL.set_ylabel(metrics)
    axL.set_xlabel('epoch')
    axL.legend(['train', 'test'], loc='upper left')

    # [右側] lossについてのグラフ
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
