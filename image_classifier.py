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
from libs.utils.utils import printWithDate

# colabとdriveの同期待ちのため
from time import sleep

# GPU使用量の調整
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.backend.tensorflow_backend import clear_session

# folder関連
from libs.utils.folder import folder_create, folder_delete

# 分割(層化k分割の交差検証)
from libs.k_fold_split import stratified_k_fold

# 評価用データの作成および読みこみ
# train/00_normal/画像ファイル)
# train/01_Gla/(画像ファイル)

# モデルコンパイル
from keras.optimizers import SGD
from libs.models import Models
from libs.utils.model import model_compile

# 訓練用データの作成およびデータ拡張後の読みこみ
from libs.data_generator import Training, Validation

# modelの定義およびコンパイル、学習、保存、学習経過のプロット
from libs.learning import Learning, plot_hist

# 評価、結果の分析
from libs.utils.model import model_load, model_delete
from libs.auc_analysis import Miss_classify
from libs.auc_analysis import cross_making, miss_summarize
from libs.auc_analysis import (summary_analysis_binary,
                               summary_analysis_categorical)

# configparserを使った設定ファイルの読み込み
import configparser
from libs.utils.utils import check_options

import libs.error as error

# pandasを使っcsvファイルの読み込みに対応
import pandas as pd


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
    if os.path.isfile('options.conf') is False:
        error.option_file_not_exist()
    options = configparser.ConfigParser()
    options.optionxform = str  # キーをcase-sensitiveにする
    options.read('options.conf', encoding='utf-8')
    check_options(options)

    # 変数の整形
    image_size = [options.getint('ImageSize', 'width'),
                  options.getint('ImageSize', 'height')]

    # 設定ファイルで指定したcsvファイルを読み込み
    df = pd.read_csv(options['CSV']['csv_filename'])

    # 分類ラベルをリスト化し、リストの長さを調べて分類数とする
    label_list = df[options['CSV']['label_column']].unique().tolist()
    classes = len(label_list)
    printWithDate(f'{classes} classes found')

    # 2値分類の場合
    if classes == 2:
        positive_label = str(options['Analysis']['positive_label'])
        if positive_label not in label_list:
            # 分類ラベルにpositive_labelが無い場合、エラーを出して終了する
            error.positive_label_not_found(positive_label)
        for label in label_list:
            if str(label) == positive_label:
                printWithDate(f'positive label is \"{label}\".')
            else:
                printWithDate(f'negative label is \"{label}\".')
                negative_label = str(label)
        # negative_labelが先(0), positive_labelが後(1)に来るようにlabel_listを上書きする
        label_list = [negative_label, positive_label]

    # pic_modeを決める
    if classes == 2:
        PIC_MODE = 0
    else:
        PIC_MODE = 1

    # ここで、データ拡張の方法を指定。
    train_num_mode_dic = {}
    # gradeごとにデータ拡張の方法を変える場合はここを変更
    for label in label_list:
        train_num_mode_dic[label] = [options.getint('DataGenerate', 'num_of_augs'),
                                     options.getboolean('DataGenerate', 'use_flip')]

    # 層化k分割
    if options['CSV']['ID_column'] == "None":
        hasID = False
    else:
        hasID = True
    printWithDate(f"hasID = {hasID}")

    printWithDate("spliting dataset")
    df_train_list, df_test_list = \
        stratified_k_fold(options.getint('Validation', 'k'),
                          options['CSV'],
                          options['FolderName']['split_info'],
                          df, hasID)

    # 分割ごとに
    for idx in range(options.getint('Validation', 'k')):
        printWithDate("processing sprited dataset",
                      f"{idx + 1}/{options.getint('Validation', 'k')}")

        # 評価用データについて
        printWithDate("making data for validation",
                      f"[{idx + 1}/{options.getint('Validation', 'k')}]")
        validation = Validation(image_size,
                                options['FolderName'],
                                classes, options['Analysis']['positive_label'],
                                PIC_MODE, idx, df_test_list[idx])
        validation.pic_df_test()
        X_val, y_val, W_val = validation.pic_gen_data()

        # 訓練用データについて
        printWithDate("making data for training",
                      f"[{idx + 1}/{options.getint('Validation', 'k')}]")
        training = Training(options['FolderName'], idx, PIC_MODE,
                            train_num_mode_dic,
                            image_size, classes,
                            options['Analysis']['positive_label'],
                            options['ImageDataGenerator'],
                            options.getint('HyperParameter', 'batch_size'),
                            df_train_list[idx])
        training.pic_df_training()

        # model定義
        # modelの関係をLearningクラスのコンストラクタで使うから先にここで定義
        for model_name, is_use in options['NetworkUsing'].items():
            if is_use == 'False':  # valueはstr型
                continue

            set_session(tf.Session(config=config))

            folder_create(model_name)
            history_folder = os.path.join(model_name, "history")
            model_folder = os.path.join(model_name, "model")
            miss_folder = os.path.join(model_name, "miss")
            # roc_folder = os.path.join(model_name, "roc")
            # result_file = os.path.join(model_name, "result.csv")
            summary_file = os.path.join(model_name, "summary.csv")
            cross_file = os.path.join(model_name, "cross.csv")
            miss_file = os.path.join(model_name, "miss_summary.csv")
            # "VGG16","VGG19","DenseNet121","DenseNet169","DenseNet201",
            # "InceptionResNetV2","InceptionV3","ResNet50","Xception"
            model_ch = Models(image_size, classes, PIC_MODE)

            if model_name == 'VGG16':
                model = model_ch.vgg16()
            elif model_name == 'VGG19':
                model = model_ch.vgg19()
            elif model_name == 'DenseNet121':
                model = model_ch.dense121()
            elif model_name == 'DenseNet169':
                model = model_ch.dense169()
            elif model_name == 'DenseNet201':
                model = model_ch.dense201()
            elif model_name == 'InceptionResNetV2':
                model = model_ch.inception_resnet2()
            elif model_name == 'InceptionV3':
                model = model_ch.inception3()
            elif model_name == 'ResNet50':
                model = model_ch.resnet50()
            elif model_name == 'Xception':
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
            learning = Learning(options['FolderName'], idx, PIC_MODE,
                                train_num_mode_dic,
                                image_size, classes,
                                options['Analysis']['positive_label'],
                                options['ImageDataGenerator'],
                                options.getint('HyperParameter', 'batch_size'),
                                model_folder, model, X_val, y_val,
                                options.getint('HyperParameter', 'epochs'),
                                df_train_list[idx])

            # 訓練実行
            history = learning.learning_model()
            printWithDate(
                f"Learning finished [{idx + 1}/{options.getint('Validation', 'k')}]")

            plot_hist(history, history_folder, idx)
            model_load(model, model_folder, idx)
            y_pred = model.predict(X_val)
            Miss_classify(idx, y_pred, y_val, W_val,
                          miss_folder, label_list).miss_csv_making()

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
    for model_name, is_use in options['NetworkUsing'].items():
        if is_use == 'False':  # valueはstr型
            continue

        miss_folder = os.path.join(model_name, "miss")
        summary_file = os.path.join(model_name, "summary.csv")
        cross_file = os.path.join(model_name, "cross.csv")
        miss_file = os.path.join(model_name, "miss_summary.csv")
        fig_file = os.path.join(model_name, "figure.png")
        # miss_summary.csvを元に各種解析を行う
        # Kは不要
        miss_summarize(miss_folder, miss_file)
        cross_making(miss_folder, options.getint(
            'Validation', 'k'), cross_file)
        if PIC_MODE == 0:
            summary_analysis_binary(miss_file, summary_file, fig_file,
                                    label_list,
                                    str(options['Analysis']['positive_label']),
                                    options.getfloat('Analysis', 'alpha'))
        elif PIC_MODE == 1:
            summary_analysis_categorical(miss_file, summary_file,
                                         label_list,
                                         options.getfloat('Analysis', 'alpha'))

    printWithDate("main() function is end")
    return


if __name__ == '__main__':
    main()
