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

# 分割(グループ層化k分割の交差検証)
from libs.k_fold_split import Stratified_group_k_fold

# モデルコンパイル
from tensorflow.keras.optimizers import Adam
from libs.models import Models

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

    # 層化k分割
    if options['CSV']['ID_column'] == "None":
        hasID = False
    else:
        hasID = True
    printWithDate(f"hasID = {hasID}")

    printWithDate("spliting dataset")
    sgkf = Stratified_group_k_fold(csv_config=options['CSV'],
                                   n_splits=options.getint('Validation', 'k'),
                                   shuffle=True, random_state=42,
                                   split_info_folder=
                                   options['FolderName']['split_info'])
    df_train_list, df_test_list = sgkf.k_fold_regressor(df)

    # 分割ごとに
    for idx in range(options.getint('Validation', 'k')):
        printWithDate("processing sprited dataset",
                      f"{idx + 1}/{options['Validation']['k']}")

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
            learning = Learning(
                directory=options['FolderName']['dataset'],
                csv_config=options['CSV'],
                df_train=df_train_list[idx], df_validation=df_test_list[idx],
                label_list=label_list, idx=idx,
                base_augmentation=dict(options['BaseImageAugmentation']),
                extra_augmentation=dict(options['ExtraImageAugmentation']),
                image_size=image_size, classes=classes,
                batch_size=options.getint('HyperParameter', 'batch_size'),
                model_folder=model_folder,
                epochs=options.getint('HyperParameter', 'epochs'))

            # 訓練実行
            history = learning.train(model)
            printWithDate(
                f"Learning finished [{idx + 1}/{options.getint('Validation', 'k')}]")

            plot_hist(history, history_folder, metrics, idx)
            model_load(model, model_folder, idx)
            W_val, y_val, y_pred = learning.predict(model)

            Miss_regression(idx, y_pred, y_val, W_val,
                            miss_folder).miss_csv_making()
            printWithDate(
                f"Analysis finished [{idx + 1}/{options.getint('Validation', 'k')}]")
            model_delete(model, model_folder, idx)

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
