#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 回帰

import os
import sys
from tqdm import trange
from libs.utils.utils import printWithDate

# colabとdriveの同期待ちのため
from time import sleep

# GPU使用量の調整
import tensorflow as tf
from tensorflow.keras.backend import clear_session

# folder関連
from libs.utils.folder import folder_create, folder_delete

# 分割(層化k分割の交差検証)
from libs.k_fold_split import simple_k_fold

# 評価用データの作成および読みこみ
# train/00_normal/画像ファイル)
# train/01_Gla/(画像ファイル)

# モデルコンパイル
from tensorflow.keras.optimizers import Adam
from libs.models import Models

# 訓練用データの作成およびデータ拡張後の読みこみ
from libs.data_generator import Training, Validation

# modelの定義およびコンパイル、学習、保存、学習経過のプロット
from libs.learning import Learning, plot_hist

# 評価、結果の分析
from libs.utils.model import model_load, model_delete
from libs.auc_analysis import Miss_regression
from libs.auc_analysis import miss_summarize
from libs.auc_analysis import summary_analysis_regression

# configparserを使った設定ファイルの読み込み
import configparser
from libs.utils.utils import check_options

import libs.error as error

# pandasを使っcsvファイルの読み込みに対応
import pandas as pd

PIC_MODE = 2


def main():
    printWithDate("main() function is started")

    # GPUに過負荷がかかると実行できなくなる。∴余力を持たしておく必要がある。
    # 必要最小限のメモリを確保する
    # GPUが無いとエラー
    # GPUはひとつだけ使用する
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if len(gpus) > 0:
         tf.config.experimental.set_memory_growth(gpus[0], True)
         tf.device(gpus[0][0])
    else:
        cpus = tf.config.experimental.list_physical_devices('CPU')
        tf.device(cpus[0][0])

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
    # TODO: PIC_MODEを廃止して
    #       classes=1: 回帰, classes=2: 2値分類, classes>=3: 多値分類
    #       と扱うようにする
    label_list = ["regression"]
    classes = len(label_list)
    printWithDate(f'regression mode')

    # ここで、データ拡張の方法を指定。
    # TODO: 回帰の目標値によって増強方法を変更出来るようにする
    #       例) 視力0.0〜0.8は[6,1]、視力0.8〜は[3,1]
    train_num_mode_dic = {}
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
        simple_k_fold(options.getint('Validation', 'k'),
                      options['CSV'],
                      options['FolderName']['split_info'],
                      df, hasID)

    # 分割ごとに
    for idx in range(options.getint('Validation', 'k')):
        printWithDate("processing sprited dataset",
                      f"{idx + 1}/{options['Validation']['k']}")

        # 評価用データについて
        printWithDate("making data for validation",
                      f"[{idx + 1}/{options.getint('Validation', 'k')}]")
        validation = Validation(image_size,
                                options['FolderName'],
                                classes, options['Analysis']['positive_label'],
                                PIC_MODE, idx, df_test_list[idx])
        validation.pic_df_test_reg()
        X_val, y_val, W_val = validation.pic_gen_data_reg()

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
        training.pic_df_training_reg()

        # model定義
        # modelの関係をLearningクラスのコンストラクタで使うから先にここで定義
        for model_name, is_use in options['NetworkUsing'].items():
            if is_use == 'False':  # valueはstr型
                continue

            clear_session()

            folder_create(model_name)
            history_folder = os.path.join(model_name, "history")
            model_folder = os.path.join(model_name, "model")
            miss_folder = os.path.join(model_name, "miss")
            # roc_folder = os.path.join(model_name, "roc")
            # result_file = os.path.join(model_name, "result.csv")
            summary_file = os.path.join(model_name, "summary.csv")
            miss_file = os.path.join(model_name, "miss_summary.csv")
            # "VGG16","VGG19","DenseNet121","DenseNet169","DenseNet201",
            # "InceptionResNetV2","InceptionV3","ResNet50","Xception"
            model_ch = Models(image_size, classes, PIC_MODE)
            model = model_ch.choose(model_name)

            # optimizerはAdam
            optimizer = Adam(lr=0.0001)

            # lossは画像解析のモードによる。
            loss = "mean_squared_error"
            metrics = 'mean_absolute_error'

            # modelをcompileする。
            model.compile(loss=loss, optimizer=optimizer, metrics=[metrics])
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

            plot_hist(history, history_folder, metrics, idx)
            model_load(model, model_folder, idx)
            y_pred = model.predict(X_val)

            Miss_regression(idx, y_pred, y_val, W_val,
                            miss_folder).miss_csv_making()
            printWithDate(
                f"Analysis finished [{idx + 1}/{options.getint('Validation', 'k')}]")
            model_delete(model, model_folder, idx)

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
        miss_file = os.path.join(model_name, "miss_summary.csv")
        fig_file = os.path.join(model_name, "figure.png")
        # miss_summary.csvを元に各種解析を行う
        # Kは不要
        miss_summarize(miss_folder, miss_file)

        summary_analysis_regression(miss_file, summary_file, fig_file)

    printWithDate("main() function is end")
    return


if __name__ == '__main__':
    main()
