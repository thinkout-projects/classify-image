#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 回帰
# データセットの準備
#   imgフォルダに適当な名前("_"禁止)で1フォルダだけ作成し、その中にすべての画像を入れる
#   画像のファイル名は"{ターゲット}_{元々のファイル名}"に修正しておく

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
from libs.utils.folder import folder_create, folder_delete, folder_clean

# 分割(層化k分割の交差検証)
from libs.k_fold_split import Split

# 評価用データの作成および読みこみ
# train/00_normal/画像ファイル)
# train/01_Gla/(画像ファイル)

# モデルコンパイル
from keras.optimizers import Adam
from libs.models import Models
from libs.utils.model import model_compile

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

    # desktop.iniの削除
    folder_clean(options['FolderName']['dataset'])

    # 設定ファイルで指定したcsvファイルを読み込み
    df = pd.read_csv(options['CSV']['csv_filename'])
    # 設定ファイルでk-foldのときに個人IDを区別しないと指定されたとき、
    # filename列を複製して個人ID列を追加
    if options['etc']['distinguishUniqueID'] == False:
        df[options['CSV']['csv_column_ID']]\
            = df[options['CSV']['csv_column_filename']].split('.')[0]

    # 分類数を調べる。
    class_list = df[options['CSV']['csv_column_label']].unique().tolist()
    classes = len(class_list)
    printWithDate(f'{classes} classes found')

    # ここで、データ拡張の方法を指定。
    train_num_mode_dic = {}
    # gradeごとにデータ拡張の方法を変える場合はここを変更
    for class_name in class_list:
        train_num_mode_dic[class_name] = [options.getint('DataGenerate', 'num_of_augs'),
                                          options.getboolean('DataGenerate', 'use_flip')]

    # 分割
    printWithDate("spliting dataset")
    split = Split(options.getint('Validation', 'k'),
                  options['CSV'],
                  options['FolderName']['split_info'],
                  df, classes, train_num_mode_dic)

    # 各列が各Splitに対応しているファイル名が列挙されたデータフレーム
    df_train, df_test = split.k_fold_split()

    # 分割ごとに
    for idx in range(options.getint('Validation', 'k')):
        printWithDate("processing sprited dataset",
                      f"{idx + 1}/{options['Validation']['k']}")

        # 評価用データについて
        printWithDate("making data for validation",
                      f"[{idx + 1}/{options.getint('Validation', 'k')}]")
        validation = Validation(image_size,
                                options['FolderName'],
                                classes, PIC_MODE, idx)
        validation.pic_df_test()
        X_val, y_val, W_val = validation.pic_gen_data()

        # 訓練用データについて
        printWithDate("making data for training",
                      f"[{idx + 1}/{options.getint('Validation', 'k')}]")
        training = Training(options['FolderName'], idx, PIC_MODE,
                            train_num_mode_dic,
                            image_size, classes,
                            options['ImageDataGenerator'],
                            options.getint('HyperParameter', 'batch_size'))
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

            # optimizerはAdam
            optimizer = Adam(lr=0.0001)

            # lossは画像解析のモードによる。
            loss = "mean_squared_error"

            # modelをcompileする。
            model_compile(model, loss, optimizer)
            learning = Learning(options['FolderName'], idx, PIC_MODE,
                                train_num_mode_dic,
                                image_size, classes,
                                options['ImageDataGenerator'],
                                options.getint('HyperParameter', 'batch_size'),
                                model_folder, model, X_val, y_val,
                                options.getint('HyperParameter', 'epochs'))

            # 訓練実行
            history = learning.learning_model()
            printWithDate(
                f"Learning finished [{idx + 1}/{options.getint('Validation', 'k')}]")

            plot_hist(history, history_folder, idx)
            model_load(model, model_folder, idx)
            y_pred = model.predict(X_val)

            Miss_regression(idx, y_pred, y_val, W_val,
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
