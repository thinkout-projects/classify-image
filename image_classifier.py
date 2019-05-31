#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 2値分類
# 解析結果
#   AUC、感度・特異度
#
# 多クラス分類
# 解析結果
#   正答率のみ

import os
import sys
from tqdm import trange
from utils import printWithDate

# colabとdriveの同期待ちのため
from time import sleep

# GPU使用量の調整
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.backend.tensorflow_backend import clear_session

# folder関連
from utils import folder_create, folder_delete, folder_clean

# 分割(層化k分割の交差検証)
from k_fold_split import Split

# 評価用データの作成および読みこみ
# train/00_normal/画像ファイル)
# train/01_Gla/(画像ファイル)

# モデルコンパイル
from keras.optimizers import SGD
from models import Models
from utils import model_compile

# 訓練用データの作成およびデータ拡張後の読みこみ
from data_generator import Training, Validation

# modelの定義およびコンパイル、学習、保存、学習経過のプロット
from learning import Learning, plot_hist

# 評価、結果の分析
from utils import model_load, model_delete
from auc_analysis import Miss_classify
from auc_analysis import cross_making, miss_summarize
from auc_analysis import (summary_analysis_binary,
                          summary_analysis_categorical)

# configparserを使った設定ファイルの読み込み
import configparser
from utils import check_options


def main():
    printWithDate("main() function is started")

    # GPUに過負荷がかかると実行できなくなる。∴余力を持たしておく必要がある。
    # 50％のみを使用することとする
    config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.8
    config.gpu_options.allow_growth = True

    # 作業ディレクトリを自身のファイルのディレクトリに変更
    os.chdir(os.path.dirname(os.path.abspath(sys.argv[0])))

    # 設定ファイルのロード
    options = configparser.ConfigParser()
    options.read('options.conf')
    check_options(options)

    # desktop.iniの削除
    folder_clean(options['FolderName']['dataset'])

    # 分類数を調べる。
    classes = len(os.listdir(options['FolderName']['dataset']))
    printWithDate(f'{classes} classes found')

    # pic_modeを決める
    if classes == 2:
        PIC_MODE = 0
    else:
        PIC_MODE = 1

    # ここで、データ拡張の方法を指定。
    folder_list = os.listdir(options['FolderName']['dataset'])
    train_num_mode_dic = {}

    # gradeごとにデータ拡張の方法を変える場合はここを変更
    for i, folder in enumerate(folder_list):
        train_num_mode_dic[folder] = [options.getint('DataGenerate', 'num_of_augs'),
                                      options.getboolean('DataGenerate', 'use_flip')]

    # 分割
    printWithDate("spliting dataset")
    split = Split(options.getint('Validation', 'k'),
                  options['FolderName']['dataset'],
                  options['FolderName']['split_info'])
    split.k_fold_split_unique()

    # 分割ごとに
    for idx in range(options.getint('Validation', 'k')):
        printWithDate("processing sprited dataset",
                      f"{idx + 1}/{options.getint('Validation', 'k')}")

        # 評価用データについて
        printWithDate("making data for validation",
                      f"[{idx + 1}/{options.getint('Validation', 'k')}]")
        validation = Validation([options.getint('ImageSize', 'width'),
                                 options.getint('ImageSize', 'height')],
                                options['FolderName']['dataset'],
                                options['FolderName']['test'],
                                options['FolderName']['split_info'],
                                classes, PIC_MODE, idx)
        validation.pic_df_test()
        X_val, y_val, W_val = validation.pic_gen_data()

        # 訓練用データについて
        printWithDate("making data for training",
                      f"[{idx + 1}/{options.getint('Validation', 'k')}]")
        training = Training(options['FolderName']['dataset'],
                            options['FolderName']['split_info'],
                            options['FolderName']['train'], idx, PIC_MODE,
                            train_num_mode_dic,
                            [options.getint('ImageSize', 'width'),
                             options.getint('ImageSize', 'height')],
                            classes,
                            options.getint('ImageDataGenerator', 'ratation_range'),
                            options.getfloat('ImageDataGenerator', 'width_shift_range'),
                            options.getfloat('ImageDataGenerator', 'height_shift_range'),
                            options.getint('ImageDataGenerator', 'shear_range'),
                            options.getfloat('ImageDataGenerator', 'zoom_range'),
                            options.getint('HyperParameter', 'batch_size'))
        training.pic_df_training()

        # model定義
        # modelの関係をLearningクラスのコンストラクタで使うから先にここで定義
        for output_folder in options['NetworkUsing']:
            # TODO: for文のoptions['NetworkUsing']対応
            set_session(tf.Session(config=config))

            folder_create(output_folder)
            history_folder = os.path.join(output_folder, "history")
            model_folder = os.path.join(output_folder, "model")
            miss_folder = os.path.join(output_folder, "miss")
            # roc_folder = os.path.join(output_folder, "roc")
            # result_file = os.path.join(output_folder, "result.csv")
            summary_file = os.path.join(output_folder, "summary.csv")
            cross_file = os.path.join(output_folder, "cross.csv")
            miss_file = os.path.join(output_folder, "miss_summary.csv")
            # "VGG16","VGG19","DenseNet121","DenseNet169","DenseNet201",
            # "InceptionResNetV2","InceptionV3","ResNet50","Xception"
            model_ch = Models([options.getint('ImageSize', 'width'),
                               options.getint('ImageSize', 'height')],
                              classes, PIC_MODE)

            if output_folder == 'VGG16':
                model = model_ch.vgg16()
            elif output_folder == 'VGG19':
                model = model_ch.vgg19()
            elif output_folder == 'DenseNet121':
                model = model_ch.dense121()
            elif output_folder == 'DenseNet169':
                model = model_ch.dense169()
            elif output_folder == 'DenseNet201':
                model = model_ch.dense201()
            elif output_folder == 'InceptionResNetV2':
                model = model_ch.inception_resnet2()
            elif output_folder == 'InceptionV3':
                model = model_ch.inception3()
            elif output_folder == 'ResNet50':
                model = model_ch.resnet50()
            elif output_folder == 'Xception':
                model = model_ch.xception()

            # optimizerはSGD
            optimizer = SGD(lr=0.0001, decay=1e-6, momentum=0.9,
                            nesterov=True)  # Adam(lr = 0.0005)

            # lossは画像解析のモードによる。
            if PIC_MODE == 0:
                loss = "binary_crossentropy"
            elif PIC_MODE == 1:
                loss = "categorical_crossentropy"

            # modelをcompileする。
            model_compile(model, loss, optimizer)
            learning = Learning(options['FolderName']['dataset'],
                                options['FolderName']['dataset_info'],
                                options['FolderName']['train'], idx, PIC_MODE,
                                train_num_mode_dic,
                                [options.getint('ImageSize', 'width'),
                                    options.getint('ImageSize', 'height')],
                                classes,
                                options.getint('ImageDataGenerator', 'ratation_range'),
                                options.getfloat('ImageDataGenerator', 'width_shift_range'),
                                options.getfloat('ImageDataGenerator', 'height_shift_range'),
                                options.getint('ImageDataGenerator', 'shear_range'),
                                options.getfloat('ImageDataGenerator', 'zoom_range'),
                                options.getint('HyperParameter', 'batch_size'),
                                model_folder, model,
                                X_val, y_val,
                                options.getint('HyperParameter', 'epochs'))

            # 訓練実行
            history = learning.learning_model()
            printWithDate(
                f"Learning finished [{idx + 1}/{options.getint('Validation', 'k')}]")

            plot_hist(history, history_folder, idx)
            model_load(model, model_folder, idx)
            y_pred = model.predict(X_val)
            Miss_classify(idx, y_pred, y_val, W_val,
                          options['FolderName']['test'],
                          miss_folder).miss_csv_making()

            printWithDate(
                f"Analysis finished [{idx + 1}/{options.getint('Validation', 'k')}]")
            model_delete(model, model_folder, idx)
            clear_session()

        # 訓練用フォルダおよびテスト用フォルダを削除する。
        folder_delete(options['FolderName']['train'])
        folder_delete(options['FolderName']['test'])

        # colabとdriveの同期待ちをする
        for i in trange(options.getint('etc', 'wait_sec'),
                        desc='Waiting for syncing with GDrive'):
            sleep(1)

    printWithDate("output Summary Analysis")
    for output_folder in options['NetworkUsing']:
        # TODO: for文のoptions['NetworkUsing']対応
        miss_folder = os.path.join(output_folder, "miss")
        summary_file = os.path.join(output_folder, "summary.csv")
        cross_file = os.path.join(output_folder, "cross.csv")
        miss_file = os.path.join(output_folder, "miss_summary.csv")
        fig_file = os.path.join(output_folder, "figure.png")
        # miss_summary.csvを元に各種解析を行う
        # Kは不要
        miss_summarize(miss_folder, miss_file)
        cross_making(miss_folder, options.getint('Validation', 'k'), cross_file)
        if PIC_MODE == 0:
            summary_analysis_binary(miss_file, summary_file, fig_file,
                                    options['FolderName']['dataset'],
                                    options.getfloat('Validation', 'alpha'))
        elif PIC_MODE == 1:
            summary_analysis_categorical(miss_file, summary_file,
                                         options['FolderName']['dataset'],
                                         options.getfloat('Validation', 'alpha'))

    printWithDate("main() function is end")
    return


if __name__ == '__main__':
    main()
