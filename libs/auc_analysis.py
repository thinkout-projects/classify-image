#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 2値分類、他クラス分類に関する結果表示(csv出力もする)

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
from scipy import stats
from .utils.utils import clopper_pearson
from .utils.folder import folder_create
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc

# モニターのないサーバーでも動作するように
plt.switch_backend('agg')


class Miss_classify(object):
    def __init__(self, idx, y_pred, y_val, W_val, miss_folder, label_list):
        self.idx = idx
        # 2値分類のとき、y_pred, y_valは、陰性が0, 陽性が1を示す
        self.y_pred = y_pred
        self.y_val = y_val
        self.W_val = W_val
        self.class_list = label_list
        self.miss_folder = miss_folder

    def miss_detail(self):
        # pred_listは[0, 1, 0, 1]のように示されるAI解答番号リスト
        pred_list = []
        # true_listは[0, 1, 0, 0]のように示される正解番号リスト
        true_list = []
        # class_score_dicは、
        # class名をキーとしてAI解答リスト([0.1, 0.9], [0.8, 0.2])を値に持つ辞書
        class_score_dic = {}
        for class_name in self.class_list:
            class_score_dic[class_name] = []

        for pred, true in zip(self.y_pred, self.y_val):
            # pred_ansはfileごとの確率の羅列
            pred_ans = pred.argmax()
            # pred_listはすべての問題のAIによる回答
            pred_list.append(pred_ans)

            true_list.append(true)

            for idx, class_name in enumerate(self.class_list):
                class_score_dic[class_name].append(pred[idx])
        return pred_list, true_list, class_score_dic

    def miss_csv_making(self):
        '''
        全てのファイルをフォルダごとにcsvファイルに書き込む
        '''
        pred_list, true_list, class_score_dic = \
            Miss_classify.miss_detail(self)

        # missフォルダ作成
        folder_create(self.miss_folder)

        # クラス分の[]を用意する。
        file_name_list = []
        for num in range(len(pred_list)):
            # y_val[num]は当該ファイルの正答のベクトル、argmaxで正答番号がわかる
            # W_val[num]は当該ファイルのパス
            file_name_list.append(os.path.basename(self.W_val[num]))

        df = pd.DataFrame()
        df["filename"] = file_name_list
        df["true"] = true_list
        df["predict"] = pred_list

        for class_name, score_list in class_score_dic.items():
            df[str(class_name)] = score_list
        miss_file = "miss" + "_" + str(self.idx) + "." + "csv"
        miss_fpath = os.path.join(self.miss_folder, miss_file)
        df.to_csv(miss_fpath, index=False, encoding="utf-8")
        return


class Miss_regression(object):
    def __init__(self, idx, y_pred, y_val, W_val, miss_folder):
        self.idx = idx
        self.y_pred = y_pred
        self.y_val = y_val
        self.W_val = W_val
        self.miss_folder = miss_folder

    def miss_csv_making(self):
        # missフォルダ作成
        folder_create(self.miss_folder)

        # クラス分の[]を用意する。
        file_name_list = []
        for W in self.W_val:
            file_name_list.append(os.path.basename(W))

        df = pd.DataFrame()
        df["filename"] = file_name_list
        df["true"] = self.y_val
        df["predict"] = self.y_pred

        miss_file = "miss" + "_" + str(self.idx) + "." + "csv"
        miss_fpath = os.path.join(self.miss_folder, miss_file)
        df.to_csv(miss_fpath, index=False, encoding="utf-8")
        return


def cross_making(miss_folder, k, cross_file):
    true_list = []
    predict_list = []
    df_pre_cross = pd.DataFrame(columns=["true", "predict"])
    for i in range(k):
        csv_file = "miss_" + str(i) + ".csv"
        csv_fpath = os.path.join(miss_folder, csv_file)
        df = pd.read_csv(csv_fpath, encoding="utf-8")
        true = df["true"]
        predict = df["predict"]
        true_list.extend(true)
        predict_list.extend(predict)

    df_pre_cross["true"] = true_list
    df_pre_cross["predict"] = predict_list
    df_cross = pd.crosstab(
        df_pre_cross["predict"], df_pre_cross["true"], margins=True)
    df_cross.to_csv(cross_file, encoding="utf-8")
    return


def miss_summarize(miss_folder, miss_file):
    df2 = pd.DataFrame()
    for csv_file in os.listdir(miss_folder):
        csv_fpath = os.path.join(miss_folder, csv_file)
        df = pd.read_csv(csv_fpath, encoding="utf-8")
        df2 = df2.append(df)
    df2.to_csv(miss_file, encoding="utf-8", index=False)
    return


def summary_analysis_binary(miss_summary_file, summary_file, roc_fig,
                            positive_label, alpha):
    df = pd.read_csv(miss_summary_file, encoding="utf-8")
    # libs/utils/utils.pyのfpath_tag_making()により、
    # 陰性は0, 陽性は1のタグが付けられることは確定している
    df0 = df[df["true"] == 0]
    df1 = df[df["true"] == 1]
    n_normal = len(df0)
    n_des = len(df1)

    # y_true, y_predはsklearn.metrics.roc_auc_scoreのy_true, y_scoreに渡される
    y_true = np.array(df["true"])
    y_pred = np.array(df[str(positive_label)])

    # AUCについて
    # y_pred, y_trueを用いて95%信頼区間を求める
    AUCs = roc_auc_ci(y_true, y_pred, alpha, positive=1)
    print("AUC")
    print(AUCs)

    # 感度1について
    k_des = len(df1[df1["predict"] == 1])
    sensitivity = float(k_des/n_des)
    sensitivity_low, sensitivity_up = clopper_pearson(k_des, n_des, alpha)
    sensitivities = [sensitivity, sensitivity_low, sensitivity_up]
    print("感度")
    print(sensitivities)

    # 特異度1について
    k_normal = len(df0[df0["predict"] == 0])
    specificity = float(k_normal/n_normal)
    specificity_low, specificity_up = clopper_pearson(
        k_normal, n_normal, alpha)
    specificities = [specificity, specificity_low, specificity_up]
    print("特異度")
    print(specificities)

    df_out = pd.DataFrame()
    df_out["AUC"] = AUCs
    df_out["sensitivity"] = sensitivities
    df_out["specificity"] = specificities

    df_out.to_csv(summary_file, index=False, encoding="utf-8")

    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, linewidth=3,
             label='ROC curve (area = %0.3f)' % roc_auc)
    plt.scatter(np.array(1-specificity),
                np.array(sensitivity), s=50, c="green")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc="lower right")
    plt.savefig(roc_fig)
    plt.close()
    return


def roc_auc_ci(y_true, y_score, alpha, positive=1):
    AUC = roc_auc_score(y_true, y_score)
    # 有病グループの数がN1, 正常群の数がN2
    N1 = sum(y_true == positive)
    N2 = sum(y_true != positive)
    # Q1は
    Q1 = AUC / (2 - AUC)
    Q2 = 2*AUC**2 / (1 + AUC)
    SE_AUC = sqrt((AUC*(1 - AUC) + (N1 - 1)*(Q1 - AUC**2) +
                   (N2 - 1)*(Q2 - AUC**2)) / (N1*N2))
    a, b = stats.norm.interval(alpha, loc=0, scale=1)
    lower = AUC + a*SE_AUC
    upper = AUC + b*SE_AUC
    if lower < 0:
        lower = 0
    if upper > 1:
        upper = 1
    return [AUC, lower, upper]


def summary_analysis_categorical(miss_summary_file, summary_file, label_list,
                                 alpha):
    df = pd.read_csv(miss_summary_file, encoding="utf-8")
    df_out = pd.DataFrame()
    for i, label in enumerate(label_list):
        df0 = df[df["true"] == i]
        n_0 = len(df0)
        df00 = df0[df0["predict"] == i]
        k_0 = len(df00)
        accuracy = float(k_0/n_0)
        accuracy_low, accuracy_up = clopper_pearson(k_0, n_0, alpha)
        accuracies = [accuracy, accuracy_low, accuracy_up]
        print(label)
        print(accuracies)
        df_out[label] = accuracies
    df_out.to_csv(summary_file, index=False, encoding="utf-8")
    return


def summary_analysis_regression(miss_summary_file, summary_file, fig_file):
    df = pd.read_csv(miss_summary_file, encoding="utf-8")
    y_true = df["true"]
    y_pred = df["predict"]
    r, p = stats.pearsonr(y_true, y_pred)
    print("相関係数")
    print(r, p)
    df_out = pd.DataFrame()
    df_out["pearsonr"] = r, p
    df_out.to_csv(summary_file, index=False, encoding="utf-8")

    plt.figure()
    plt.scatter(y_true, y_pred)
    plt.xlabel('Ground value')
    plt.ylabel('Predict Value')
    plt.title('Prediction')
    plt.savefig(fig_file)
    plt.close()
