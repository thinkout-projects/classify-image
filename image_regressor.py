#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 機械学習テンプレートコード
# Usage: python main.py

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
from keras.optimizers import Adam, SGD
from models import Models
from utils import model_compile

# 訓練用データの作成およびデータ拡張後の読みこみ
from data_generator import Training, Validation

# modelの定義およびコンパイル、学習、保存、学習経過のプロット
from learning import Learning, plot_hist

# 評価、結果の分析
from utils import model_load, model_delete
from auc_analysis import Miss_classify, Miss_regression
from auc_analysis import cross_making, miss_summarize
from auc_analysis import (summary_analysis_binary,
                          summary_analysis_categorical,
                          summary_analysis_regression)

# 設定ファイルの読み込み
import configparser

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

    # Settingsのロード
    settings = configparser.ConfigParser()
    settings.read('options.conf')

    # desktop.iniの削除
    folder_clean(settings.get('Folders', 'IMG_ROOT'))

    # 分類数を調べる。
    classes = len(os.listdir(settings.get('Folders', 'IMG_ROOT')))
    printWithDate(f'{classes} classes found')

    # ここで、データ拡張の方法を指定。
    folder_list = os.listdir(settings.get('Folders', 'IMG_ROOT'))
    train_num_mode_dic = {}

    # gradeごとにデータ拡張の方法を変える場合はここを変更
    for i, folder in enumerate(folder_list):
        train_num_mode_dic[folder] = [settings.get('DataGenerator', 'NUM_OF_AUGS'), settings.get('DataGenerator', 'USE_FLIP')]

    # 分割
    printWithDate("spliting dataset")
    split = Split(settings.get('validation', 'K'), settings.get('Folders', 'IMG_ROOT'), settings.get('Folders', 'DATASET_FOLDER'))
    split.k_fold_split_unique()

    # 分割ごとに
    for idx in range(settings.get('validation', 'K')):
        printWithDate(f"processing sprited dataset {idx + 1}/{settings.get('validation', 'K')}")

        # 評価用データについて
        printWithDate(f"making data for validation [{idx + 1}/{settings.get('validation', 'K')}]")
        validation = Validation(settings.get('HyperParameter', 'IMG_SIZE'), settings.get('Folders', 'IMG_ROOT'),
                                settings.get('Folders', 'TEST_ROOT'), settings.get('Folders', 'DATASET_FOLDER'),
                                classes, PIC_MODE, idx)
        validation.pic_df_test()
        X_val, y_val, W_val = validation.pic_gen_data()

        # 訓練用データについて
        printWithDate(f"making data for training [{idx + 1}/{settings.get('validation', 'K')}]")
        training = Training(settings.get('Folders', 'IMG_ROOT'), settings.get('Folders', 'DATASET_FOLDER'),
                            settings.get('Folders', 'TRAIN_ROOT'), idx, PIC_MODE,
                            train_num_mode_dic, settings.get('HyperParameter', 'IMG_SIZE'),
                            classes, settings.get('HyperParameter', 'ROTATION_RANGE'),
                            settings.get('HyperParameter', 'WIDTH_SHIFT_RANGE'),
                            settings.get('HyperParameter', 'HEIGHT_SHIFT_RANGE'), settings.get('HyperParameter', 'SHEAR_RANGE'),
                            settings.get('HyperParameter', 'ZOOM_RANGE'), settings.get('HyperParameter', 'BATCH_SIZE'))
        training.pic_df_training()

        # model定義
        # modelの関係をLearningクラスのコンストラクタで使うから先に、ここで定義
        for output_folder in settings.get('Folders', 'OUTPUT_FOLDER_LIST'):
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
            model_ch = Models(settings.get('HyperParameter', 'IMG_SIZE'), classes, PIC_MODE)

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
            optimizer = Adam(lr=0.0001)

            # lossは画像解析のモードによる。
            loss = "mean_squared_error"

            # modelをcompileする。
            model_compile(model, loss, optimizer)
            learning = Learning(settings.get('Folders', 'IMG_ROOT'), settings.get('Folders', 'DATASET_FOLDER'),
                                settings.get('Folders', 'TRAIN_ROOT'), idx, PIC_MODE,
                                train_num_mode_dic, settings.get('HyperParameter', 'IMG_SIZE'), classes,
                                settings.get('HyperParameter', 'ROTATION_RANGE'),
                                settings.get('HyperParameter', 'WIDTH_SHIFT_RANGE'),
                                settings.get('HyperParameter', 'HEIGHT_SHIFT_RANGE'),
                                settings.get('HyperParameter', 'SHEAR_RANGE'), settings.get('HyperParameter', 'ZOOM_RANGE'),
                                settings.get('HyperParameter', 'BATCH_SIZE'), model_folder, model,
                                X_val, y_val, settings.get('HyperParameter', 'EPOCHS'))

            # 訓練実行
            history = learning.learning_model()
            printWithDate(f"Learning finished [{idx + 1}/{settings.get('validation', 'K')}]")

            plot_hist(history, history_folder, idx)
            model_load(model, model_folder, idx)
            y_pred = model.predict(X_val)

            Miss_regression(idx, y_pred, y_val, W_val,
                            miss_folder).miss_csv_making()
            printWithDate(f"Analysis finished [{idx + 1}/{settings.get('validation', 'K')}]")
            model_delete(model, model_folder, idx)
            clear_session()

        # 訓練用フォルダおよびテスト用フォルダを削除する。
        folder_delete(settings.get('Folders', 'TRAIN_ROOT'))
        folder_delete(settings.get('Folders', 'TEST_ROOT'))

        # colabとdriveの同期待ちをする
        for i in trange(settings.get('etc', 'WAITSEC'),
                        desc='Waiting for syncing with GDrive'):
            sleep(1)

    printWithDate("output Summary Analysis")
    for output_folder in settings.get('Folders', 'OUTPUT_FOLDER_LIST'):
        miss_folder = os.path.join(output_folder, "miss")
        summary_file = os.path.join(output_folder, "summary.csv")
        cross_file = os.path.join(output_folder, "cross.csv")
        miss_file = os.path.join(output_folder, "miss_summary.csv")
        fig_file = os.path.join(output_folder, "figure.png")
        # miss_summary.csvを元に各種解析を行う
        # Kは不要
        miss_summarize(miss_folder, miss_file)

        summary_analysis_regression(miss_file, summary_file, fig_file)

    printWithDate("main() function is end")
    return


if __name__ == '__main__':
    main()
