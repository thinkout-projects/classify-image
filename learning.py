#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from utils import folder_create

# VGG16のネットワーク系
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D

# VGG 16
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.xception import Xception
from keras.applications.inception_v3 import InceptionV3
from keras.applications.densenet import DenseNet201, DenseNet169, DenseNet121
from keras.applications.resnet50 import ResNet50

# plot用に
import matplotlib.pyplot as plt

# main関数
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from training_data import Training
from utils import fpath_tag_making
# regression用に


class Models(object):
    '''
    modelを定義するクラス
    '''
    # InceptionResNetV2
    # Xception
    # InceptionV3
    # DenseNet201
    # ResNet50

    def __init__(self, size, classes, pic_mode):
        self.ch = 3
        self.size = size
        self.w = self.size[0]
        self.h = self.size[1]
        self.classes = classes
        self.pic_mode = pic_mode

    def inception_resnet2(self):
        input_tensor = Input(shape=(self.h, self.w, self.ch))
        design_model = InceptionResNetV2(
            include_top=False, weights='imagenet', input_tensor=input_tensor)

        # FC層を構築
        top_model = Sequential()
        top_model.add(Flatten(input_shape=design_model.output_shape[1:]))
        top_model.add(Dense(256, activation='relu'))
        top_model.add(Dropout(0.5))
        if(self.pic_mode != 2):
            top_model.add(Dense(self.classes, activation='softmax'))
        else:
            top_model.add(Dense(self.classes, activation='relu'))

        # VGG16とFCを接続
        model = Model(input=design_model.input,
                      output=top_model(design_model.output))
        return model

    def xception(self):
        input_tensor = Input(shape=(self.h, self.w, self.ch))
        design_model = Xception(
            include_top=False, weights='imagenet', input_tensor=input_tensor)

        # FC層を構築
        top_model = Sequential()
        top_model.add(Flatten(input_shape=design_model.output_shape[1:]))
        top_model.add(Dense(256, activation='relu'))
        top_model.add(Dropout(0.5))
        if(self.pic_mode != 2):
            top_model.add(Dense(self.classes, activation='softmax'))
        else:
            top_model.add(Dense(self.classes, activation='relu'))

        # VGG16とFCを接続
        model = Model(input=design_model.input,
                      output=top_model(design_model.output))
        return model

    def inception3(self):
        input_tensor = Input(shape=(self.h, self.w, self.ch))
        design_model = InceptionV3(
            include_top=False, weights='imagenet', input_tensor=input_tensor)

        # FC層を構築
        top_model = Sequential()
        top_model.add(Flatten(input_shape=design_model.output_shape[1:]))
        top_model.add(Dense(256, activation='relu'))
        top_model.add(Dropout(0.5))
        if(self.pic_mode != 2):
            top_model.add(Dense(self.classes, activation='softmax'))
        else:
            top_model.add(Dense(self.classes, activation='relu'))

        # VGG16とFCを接続
        model = Model(input=design_model.input,
                      output=top_model(design_model.output))
        return model

    def dense121(self):
        input_tensor = Input(shape=(self.h, self.w, self.ch))
        design_model = DenseNet121(
            include_top=False, weights='imagenet', input_tensor=input_tensor)

        # FC層を構築
        top_model = Sequential()
        top_model.add(Flatten(input_shape=design_model.output_shape[1:]))
        top_model.add(Dense(256, activation='relu'))
        top_model.add(Dropout(0.5))
        if(self.pic_mode != 2):
            top_model.add(Dense(self.classes, activation='softmax'))
        else:
            top_model.add(Dense(self.classes, activation='relu'))

        # VGG16とFCを接続
        model = Model(input=design_model.input,
                      output=top_model(design_model.output))
        return model

    def dense169(self):
        input_tensor = Input(shape=(self.h, self.w, self.ch))
        design_model = DenseNet169(
            include_top=False, weights='imagenet', input_tensor=input_tensor)

        # FC層を構築
        top_model = Sequential()
        top_model.add(Flatten(input_shape=design_model.output_shape[1:]))
        top_model.add(Dense(256, activation='relu'))
        top_model.add(Dropout(0.5))
        if(self.pic_mode != 2):
            top_model.add(Dense(self.classes, activation='softmax'))
        else:
            top_model.add(Dense(self.classes, activation='relu'))

        # VGG16とFCを接続
        model = Model(input=design_model.input,
                      output=top_model(design_model.output))
        return model

    def dense201(self):
        input_tensor = Input(shape=(self.h, self.w, self.ch))
        design_model = DenseNet201(
            include_top=False, weights='imagenet', input_tensor=input_tensor)

        # FC層を構築
        top_model = Sequential()
        top_model.add(Flatten(input_shape=design_model.output_shape[1:]))
        top_model.add(Dense(256, activation='relu'))
        top_model.add(Dropout(0.5))
        if(self.pic_mode != 2):
            top_model.add(Dense(self.classes, activation='softmax'))
        else:
            top_model.add(Dense(self.classes, activation='relu'))

        # VGG16とFCを接続
        model = Model(input=design_model.input,
                      output=top_model(design_model.output))
        return model

    def resnet50(self):
        input_tensor = Input(shape=(self.h, self.w, self.ch))
        design_model = ResNet50(
            include_top=False, weights='imagenet', input_tensor=input_tensor)

        # FC層を構築
        top_model = Sequential()
        top_model.add(Flatten(input_shape=design_model.output_shape[1:]))
        top_model.add(Dense(256, activation='relu'))
        top_model.add(Dropout(0.5))
        if(self.pic_mode != 2):
            top_model.add(Dense(self.classes, activation='softmax'))
        else:
            top_model.add(Dense(self.classes, activation='relu'))

        # VGG16とFCを接続
        model = Model(input=design_model.input,
                      output=top_model(design_model.output))
        return model

    def vgg19(self):
        input_tensor = Input(shape=(self.h, self.w, self.ch))
        design_model = VGG19(
            include_top=False, weights='imagenet', input_tensor=input_tensor)

        # FC層を構築
        top_model = Sequential()
        top_model.add(Flatten(input_shape=design_model.output_shape[1:]))
        top_model.add(Dense(256, activation='relu'))
        top_model.add(Dropout(0.5))
        if(self.pic_mode != 2):
            top_model.add(Dense(self.classes, activation='softmax'))
        else:
            top_model.add(Dense(self.classes, activation='relu'))

        # VGG16とFCを接続
        model = Model(input=design_model.input,
                      output=top_model(design_model.output))
        return model

    def vgg16(self):
        '''
        VGG16(初期値Imagenet、非固定版)
        '''

        input_tensor = Input(shape=(self.h, self.w, self.ch))
        vgg16_model = VGG16(include_top=False,
                            weights='imagenet', input_tensor=input_tensor)

        # FC層を構築
        top_model = Sequential()
        top_model.add(Flatten(input_shape=vgg16_model.output_shape[1:]))
        top_model.add(Dense(256, activation='relu'))
        top_model.add(Dropout(0.5))
        if(self.pic_mode != 2):
            top_model.add(Dense(self.classes, activation='softmax'))
        else:
            top_model.add(Dense(self.classes, activation='relu'))

        # VGG16とFCを接続
        model = Model(input=vgg16_model.input,
                      output=top_model(vgg16_model.output))
        return model

    def test_model(self):
        '''
        laptopでも使用できる3層ネットワーク
        inputなどが正しいか評価するときに使用
        '''

        model = Sequential()
        model.add(Conv2D(32, (3, 3), padding='valid',
                         input_shape=(self.h, self.w, self.ch)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, (3, 3), padding='valid'))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (3, 3), padding='valid'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.classes))

        if(self.pic_mode != 2):
            model.add(Activation('softmax'))
        else:
            model.add(Activation('relu'))
        return model


class Learning(Training):
    '''
    Trainingのクラスをスーパークラスとして、サブクラスである学習クラスを作成
    '''

    def __init__(self, source_folder, dataset_folder, train_root, idx,
                 pic_mode, train_num_mode_dic, size, classes, rotation_range,
                 width_shift_range, height_shift_range, shear_range,
                 zoom_range, BATCH_SIZE, model_folder, model,
                 X_val, y_val, epochs):
        super().__init__(source_folder, dataset_folder, train_root, idx,
                         pic_mode, train_num_mode_dic, size, classes,
                         rotation_range, width_shift_range, height_shift_range,
                         shear_range, zoom_range, BATCH_SIZE)
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
        fpath_list, tag_array = fpath_tag_making(self.train_root, self.classes)
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
            workers=3,
            verbose=1)
        return history


def plot_hist(history, history_folder, idx):
    '''
    historyオブジェクト（model.fit_generatorから生まれる）の属性として
    .history["acc"]や.history["val_acc"]がある。
    元々のmovie名からタイトルを決定する
    '''

    folder_create(history_folder)
    history_file = "history" + "_" + str(idx) + "." + "jpg"
    fig, (axL, axR) = plt.subplots(ncols=2, figsize=(10, 4))
    L_title = "Accuracy_vs_Epoch"
    axL.plot(history.history['acc'])
    axL.plot(history.history['val_acc'])

    # grid表示（格子を表示する）
    axL.grid(True)
    axL.set_title(L_title)
    axL.set_ylabel('accuracy')
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
