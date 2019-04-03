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
from keras.backend import tensorflow_backend
from keras.backend import clear_session

# folder関連
from utils import folder_create, folder_delete, folder_clean

# 分割(層化k分割の交差検証)
from k_fold_split import Split

# 評価用データの作成および読みこみ
# train/00_normal/画像ファイル)
# train/01_Gla/(画像ファイル)
from validation_data import Validation

# モデルコンパイル
from keras.optimizers import Adam, SGD
from learning import Models
from utils import model_compile

# 訓練用データの作成およびデータ拡張後の読みこみ
from training_data import Training

# modelの定義およびコンパイル、学習、保存、学習経過のプロット
from learning import Learning, plot_hist

# 評価、結果の分析
from utils import model_load, model_delete
from auc_analysis import AnalysisBinary, AnalysisMulti
from auc_analysis import summary_analysis, cross_making, miss_summarize


def main():
    printWithDate("main() function is started")

    # GPUに過負荷がかかると実行できなくなる。∴余力を持たしておく必要がある。
    # 50％のみを使用することとする
    config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.8
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))

    # k-Foldの分割数を指定
    k = 5

    # PIC_MODE = 0
    # 処理
    #   2値分類
    # 解析結果
    #   AUC、感度・特異度

    PIC_MODE = 1
    # 処理
    #   多クラス分類
    # 解析結果
    #   正答率のみ

    # PIC_MODE = 2
    # 処理
    #   回帰
    # データセットの準備
    #   imgフォルダに適当な名前("_"禁止)で1フォルダだけ作成し、その中にすべての画像を入れる
    #   画像のファイル名は"{ターゲット}_{元々のファイル名}"に修正しておく

    # batch_sizeを指定
    batch_size = 32

    # data拡張の際の変数を指定
    rotation_range = 2
    width_shift_range = 0.01
    height_shift_range = 0.01
    shear_range = 0
    zoom_range = 0.1

    # 統計解析の信頼区間を指定
    alpha = 0.95

    # 各種フォルダ名指定
    IMG_ROOT = "img"
    DATASET_FOLDER = "dataset"
    TRAIN_ROOT = "train"
    TEST_ROOT = "test"
    # OUTPUT_FOLDER_LIST = ["VGG16","VGG19","DenseNet121","DenseNet169","DenseNet201",
    #                       "InceptionResNetV2","InceptionV3","ResNet50","Xception"]
    OUTPUT_FOLDER_LIST = ["VGG16"]

    # 画像サイズ(解像度)の指定
    size = [224, 224]

    # colabとdriveの同期を待つ時間(秒単位)
    # ローカルでこのコードを実行する場合、待つ必要はないので0を推奨
    WAITSEC = 120

    # ここから具体的な処理を開始

    # 作業ディレクトリを自身のファイルのディレクトリに変更
    os.chdir(os.path.dirname(os.path.abspath(sys.argv[0])))

    # desktop.iniの削除
    folder_clean(IMG_ROOT)

    # 分類数を調べる。
    classes = len(os.listdir(IMG_ROOT))
    printWithDate(f'{classes} classes found')

    # ここで、データ拡張の方法を指定。
    folder_list = os.listdir(IMG_ROOT)
    train_num_mode_dic = {}
    # 1881 1907 2552 1493 232

    for i, folder in enumerate(folder_list):
        train_num_mode_dic[folder] = [1, 1]
        if i == 2 or i == 3 or i == 4:
            train_num_mode_dic[folder] = [9, 1]

    # 分割
    printWithDate("spliting dataset")
    split = Split(k, IMG_ROOT, DATASET_FOLDER)
    split.k_fold_split_unique()

    # 分割ごとに
    for idx in range(k):
        printWithDate(f'processing sprited dataset {idx + 1} / {k}')

        # 評価用データについて
        printWithDate(f'making data for validation [{idx + 1}/{k}]')
        validation = Validation(size, IMG_ROOT, TEST_ROOT,
                                DATASET_FOLDER, classes, PIC_MODE, idx)
        validation.pic_df_test()
        X_val, y_val, W_val = validation.pic_gen_data()

        # 訓練用データについて
        printWithDate(f'making data for training [{idx + 1}/{k}]')
        training = Training(IMG_ROOT, DATASET_FOLDER, TRAIN_ROOT, idx, PIC_MODE,
                            train_num_mode_dic, size, classes, rotation_range,
                            width_shift_range, height_shift_range, shear_range,
                            zoom_range, batch_size)
        training.pic_df_training()

        # model定義
        # modelの関係をLearningクラスのコンストラクタで使うから先に、ここで定義
        for out_i, output_folder in enumerate(OUTPUT_FOLDER_LIST):
            folder_create(output_folder)
            history_folder = os.path.join(output_folder, "history")
            model_folder = os.path.join(output_folder, "model")
            miss_folder = os.path.join(output_folder, "miss")
            roc_folder = os.path.join(output_folder, "roc")
            result_file = os.path.join(output_folder, "result.csv")
            summary_file = os.path.join(output_folder, "summary.csv")
            cross_file = os.path.join(output_folder, "cross.csv")
            miss_file = os.path.join(output_folder, "miss_summary.csv")
            # "VGG19","DenseNet121","DenseNet169","DenseNet201",
            # "InceptionResNetV2","InceptionV3","ResNet50","Xception"
            model_ch = Models(size, classes, PIC_MODE)
            """
            if out_i == 0:
                model = model_ch.vgg16()
            elif out_i == 1:
                model = model_ch.vgg19()
            elif out_i == 2:
                model = model_ch.dense121()
            elif out_i == 3:
                model = model_ch.dense169()
            elif out_i == 4:
                model = model_ch.dense201()
            elif out_i == 5:
                model = model_ch.inception_resnet2()
            elif out_i == 6:
                model = model_ch.inception3()
            elif out_i == 7:
                model = model_ch.resnet50()
            elif out_i == 8:
                model = model_ch.xception()
            """
            if out_i == 0:
                model = model_ch.vgg16()

            # optimizerはSGD
            if(PIC_MODE != 2):
                optimizer = SGD(lr=0.0001, decay=1e-6, momentum=0.9,
                                nesterov=True)  # Adam(lr = 0.0005)
            else:
                optimizer = Adam(lr=0.0001)

            # lossは画像解析のモードによる。
            if PIC_MODE == 0:
                loss = "binary_crossentropy"
            elif PIC_MODE == 1:
                loss = "categorical_crossentropy"
            elif PIC_MODE == 2:
                loss = "mean_squared_error"

            # modelをcompileする。
            model_compile(model, loss, optimizer)
            epochs = 20
            learning = Learning(IMG_ROOT, DATASET_FOLDER, TRAIN_ROOT, idx, PIC_MODE,
                                train_num_mode_dic, size, classes, rotation_range,
                                width_shift_range, height_shift_range, shear_range,
                                zoom_range, batch_size, model_folder, model, X_val, y_val, epochs)

            # 訓練実行
            history = learning.learning_model()
            printWithDate(f'Learning finished [{idx + 1}/{k}]')

            plot_hist(history, history_folder, idx)
            model_load(model, model_folder, idx)
            if PIC_MODE == 0:
                analysis = AnalysisBinary(TRAIN_ROOT, TEST_ROOT, miss_folder,
                                          model_folder, roc_folder, result_file,
                                          model, X_val, y_val, W_val, idx)
            elif PIC_MODE == 1:
                analysis = AnalysisMulti(TRAIN_ROOT, TEST_ROOT, miss_folder,
                                         model_folder, result_file,
                                         model, X_val, y_val, W_val, idx)
            elif PIC_MODE == 2:
                analysis = AnalysisMulti(TRAIN_ROOT, TEST_ROOT, miss_folder,
                                         model_folder, result_file,
                                         model, X_val, y_val, W_val, idx)
            analysis.result_csv()
            printWithDate(f'Analysis finished [{idx + 1}/{k}]')
            model_delete(model, model_folder, idx)
            clear_session()
            tensorflow_backend.clear_session()

        # 訓練用フォルダおよびテスト用フォルダを削除する。
        folder_delete(TRAIN_ROOT)
        folder_delete(TEST_ROOT)

        # colabとdriveの同期待ちをする
        for i in trange(WAITSEC, desc='Waiting for syncing with GDrive'):
            sleep(1)

    printWithDate("output Summary Analysis")
    for output_folder in OUTPUT_FOLDER_LIST:
        miss_folder = os.path.join(output_folder, "miss")
        result_file = os.path.join(output_folder, "result.csv")
        summary_file = os.path.join(output_folder, "summary.csv")
        cross_file = os.path.join(output_folder, "cross.csv")
        miss_file = os.path.join(output_folder, "miss_summary.csv")

        if PIC_MODE == 0:
            summary_analysis(result_file, summary_file, IMG_ROOT, alpha)
        cross_making(miss_folder, k, cross_file)
        miss_summarize(miss_folder, k, miss_file)

    printWithDate("main() function is end")
    return


if __name__ == '__main__':
    main()
