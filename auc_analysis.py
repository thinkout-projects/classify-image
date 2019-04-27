#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 2値分類、他クラス分類に関する結果表示(csv出力もする)

import os
# import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
from scipy import stats
from utils import folder_create, clopper_pearson, num_count
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc


class Analysis_Basic(object):
    def __init__(self, train_root, model, X_val, y_val):
        self.train_root = train_root
        self.model = model
        self.X_val = X_val
        self.y_val = y_val

    def accuracy(self):
        '''
        正答率の表示
        '''

        score = self.model.evaluate(self.X_val, self.y_val)
        loss_model = score[0]
        accuracy_model = score[1]
        return loss_model, accuracy_model

    def predict(self):
        pre = self.model.predict(self.X_val)
        return pre


class Miss(object):
    def __init__(self, idx, y_pred, y_val, W_val, model, model_folder, test_folder, miss_folder):
        self.idx = idx
        self.y_pred = y_pred
        self.y_val = y_val
        self.W_val = W_val
        self.model = model
        self.model_folder = model_folder
        self.test_folder = test_folder
        self.miss_folder = miss_folder

    def miss_detail(self):
        '''
        missの数とmissした問題の番号が出力
        '''

        miss = 0
        # miss_list = []
        pre_list = []
        true_list = []
        folder_list = os.listdir(self.test_folder)
        nb_classes = len(folder_list)
        score_list = [[] for x in range(nb_classes)]
        # preはfileごとの確率の羅列
        # pre_listはすべての問題のAIによる回答
        for i, v in enumerate(self.y_pred):
            pre_ans = v.argmax()
            pre_list.append(pre_ans)
            ans = self.y_val[i].argmax()
            true_list.append(ans)
            for idx in range(nb_classes):
                score_list[idx].append(v[idx])
            if pre_ans != ans:
                miss += 1
        return miss, pre_list, true_list, score_list

    def miss_csv_making(self):
        '''
        全てのファイルをフォルダごとにcsvファイルに書き込む
        '''

        miss, pre_list, true_list, score_list = Miss.miss_detail(self)

        # missフォルダ作成
        folder_create(self.miss_folder)
        folder_list = os.listdir(self.test_folder)
        # nb_classes = len(folder_list)

        # クラス分の[]を用意する。
        file_name_list = []
        for num in range(len(pre_list)):
            # y_val[num]は当該ファイルの正答のベクトル、argmaxで正答番号がわかる
            # W_val[num]は当該ファイルのパス、\\で分割し-1にすることで一番最後のファイル名が分かる
            file_name_list.append(self.W_val[num].split("\\")[-1])

        df = pd.DataFrame()
        df["filename"] = file_name_list
        df["true"] = true_list
        df["predict"] = pre_list

        for i, score in enumerate(score_list):
            df[folder_list[i]] = score

        miss_file = "miss" + "_" + str(self.idx) + "." + "csv"
        miss_fpath = os.path.join(self.miss_folder, miss_file)
        df.to_csv(miss_fpath, index=False, encoding="utf-8")
        return miss


class AUC(object):
    '''
    AUC関連のデータ
    '''

    def __init__(self, idx, y_pred, y_val, model, roc_folder):
        self.idx = idx
        self.y_pred = y_pred
        self.y_val = y_val
        self.model = model
        self.roc_folder = roc_folder
        self.auc_list = self.auc_curve
        self.dim_list = self.auc_csv

    def auc_curve(self):
        '''
        AUCについての情報を算定する
        '''

        # 異常の症例を01においていると仮定している。
        y_pred = self.y_pred[:, 1]

        # y_testはloadのところで定義済み
        y_val = self.y_val[:, 1]
        fpr, tpr, thresholds = roc_curve(y_val, y_pred)

        # これがAUCの値
        roc_auc = auc(fpr, tpr)
        self.auc_list = roc_auc, fpr, tpr, thresholds
        return

    def auc_plot(self):
        '''
        auc_curaveにて算定された情報を基に作図
        '''

        roc_auc, fpr, tpr, thresholds = self.auc_list
        plt.figure()
        plt.plot(fpr, tpr, linewidth=3,
                 label='ROC curve (area = %0.3f)' % roc_auc)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC curve')
        plt.legend(loc="lower right")
        folder_create(self.roc_folder)
        roc_file = "ROC_curve_" + str(self.idx) + ".jpg"
        roc_fpath = os.path.join(self.roc_folder, roc_file)
        plt.savefig(roc_fpath)
        return

    def auc_csv(self):
        '''
        csvファイルも作成（その際に距離も算定する）
        '''

        roc_auc, fpr, tpr, thresholds = self.auc_list
        df_roc = pd.DataFrame(
            columns=["fpr", "tpr", "threshold", "dim1", "dim2"])
        df_roc["fpr"] = fpr
        df_roc["tpr"] = tpr
        df_roc["threshold"] = thresholds

        # best poing法の距離もリスト化
        dim_list1 = []
        for (x, y) in zip(fpr, tpr):
            dim1 = x ** 2 + (1 - y) ** 2
            dim_list1.append(dim1)
        df_roc["dim1"] = dim_list1

        # base line法の距離もリスト化
        dim_list2 = []
        for (x, y) in zip(fpr, tpr):
            dim2 = y - x
            dim_list2.append(dim2)
        df_roc["dim2"] = dim_list2
        roc_file = "roc" + "_" + str(self.idx) + "." + "csv"
        roc_fpath = os.path.join(self.roc_folder, roc_file)
        folder_create(self.roc_folder)
        df_roc.to_csv(roc_fpath, encoding="utf-8", index=False)
        self.dim_list = dim_list1, dim_list2
        return

    def sen_spe(self):
        roc_auc, fpr, tpr, thresholds = self.auc_list
        dim_list = self.dim_list
        sen_spe_list = []
        for i, dim in enumerate(dim_list):
            dim = np.array(dim)
            if i == 0:
                idx = dim.argmin()
            elif i == 1:
                idx = dim.argmax()
            specificity = 1 - fpr[idx]
            sensitivity = tpr[idx]
            threshold = thresholds[idx]
            sen_spe_list.append([sensitivity, specificity, threshold])
        return sen_spe_list


class AnalysisBinary(object):
    # class Analysis_Basic():
    # def __init__(self, train_root,model,X_val,y_val):
    # class Miss():
    # def __init__(self,idx,y_pred,y_val,W_val,model,model_folder,test_folder,miss_folder):
    # class AUC():
    # def __init__foler(self,idx,y_pred,y_val,model,roc_folder):
    def __init__(self, train_root, test_root, miss_folder, model_folder, roc_folder, result_file,
                 model, X_val, y_val, W_val, idx):

        self.train_root = train_root
        self.test_root = test_root
        self.result_file = result_file

        # Analysisの基本クラスコンストラクタ
        self.basic = Analysis_Basic(train_root, model, X_val, y_val)

        # Analysisの基本クラスの予測を使う
        self.y_pred = self.basic.predict()

        # Missクラス、AUCクラスコンストラクタ
        self.miss_class = Miss(
            idx, self.y_pred, y_val, W_val, model, model_folder, test_root, miss_folder)
        self.auc_class = AUC(idx, self.y_pred, y_val, model, roc_folder)
        self.idx = idx

    def analysis(self):
        # folderごとのtrainingデータの数が分かる
        train_dic = num_count(self.train_root)
        test_dic = num_count(self.test_root)

        # idxのテストデータにおけるloss,正答率が分かる
        loss_model, accuracy_model = self.basic.accuracy()

        # missの数を出力。その際に、回答一覧のｃｓｖファイルも作成
        miss = self.miss_class.miss_csv_making()

        # auc_curve→AUC、擬陽性率、真陽性率、閾値のリストをゲット
        self.auc_class.auc_curve()
        roc_auc, fpr, tpr, thresholds = self.auc_class.auc_list

        # ROC曲線を作図
        self.auc_class.auc_plot()

        # csvファイル作成、およびdim_listを作成しておく。
        self.auc_class.auc_csv()
        dim_list = self.auc_class.sen_spe()

        # best point法・・・距離が最も近いもの
        sensitivity1, specificity1, threshold1 = dim_list[0]

        # base line法・・・・距離が最も遠いもの
        sensitivity2, specificity2, threshold2 = dim_list[1]

        return train_dic, test_dic, loss_model, accuracy_model, miss, roc_auc,\
            sensitivity1, specificity1, threshold1,\
            sensitivity2, specificity2, threshold2

    def result_csv_making_0(self):
        '''
        初回に行名のみを記入するメソッド
        '''

        columns_list = []
        train_column_list = [
            "train" + "(" + x + ")" for x in os.listdir(self.train_root)]
        test_column_list = [
            "test" + "(" + x + ")" for x in os.listdir(self.test_root)]

        columns_list.extend(train_column_list)
        columns_list.extend(test_column_list)
        columns_list.extend(["loss", "accuracy", "miss", "AUC",
                             "sensitivity(bestpoint)", "specificity(bestpoint)", "threshold(bestpoint)",
                             "sensitivity(baseline)", "specificity(baseline)", "threshold(baseline)", ])
        df = pd.DataFrame(columns=[columns_list])
        df.to_csv(self.result_file, encoding="utf-8", index=False)
        return

    def result_csv_making(self):
        train_dic, test_dic, loss_model, accuracy_model, miss, roc_auc,\
            sensitivity1, specificity1, threshold1,\
            sensitivity2, specificity2, threshold2 = AnalysisBinary.analysis(self)
        df = pd.read_csv(self.result_file)
        df2 = pd.DataFrame()
        for column in train_dic.keys():
            df2["train" + "(" + column + ")"] = [train_dic[column]]
        for column in test_dic.keys():
            df2["test" + "(" + column + ")"] = [test_dic[column]]

        df2["loss"] = [loss_model]
        df2["accuracy"] = [accuracy_model]
        df2["miss"] = [miss]
        df2["AUC"] = [roc_auc]
        df2["sensitivity(bestpoint)"] = [sensitivity1]
        df2["specificity(bestpoint)"] = [specificity1]
        df2["threshold(bestpoint)"] = [threshold1]
        df2["sensitivity(baseline)"] = [sensitivity2]
        df2["specificity(baseline)"] = [specificity2]
        df2["threshold(baseline)"] = [threshold2]
        df = df.append(df2)
        df.to_csv(self.result_file, encoding="utf-8", index=False)
        return

    def result_csv(self):
        if self.idx == 0:
            AnalysisBinary.result_csv_making_0(self)
        AnalysisBinary.result_csv_making(self)
        return


# ここまで2値分類
# これ以降は多クラス分類
class AnalysisMulti(object):
    def __init__(self, train_root, test_root, miss_folder, model_folder, result_file,
                 model, X_val, y_val, W_val, idx):
        self.train_root = train_root
        self.test_root = test_root
        self.result_file = result_file

        # Analysisの基本クラスコンストラクタ
        self.basic = Analysis_Basic(train_root, model, X_val, y_val)

        # Analysisの基本クラスの予測を使う
        self.y_pred = self.basic.predict()

        # Missクラス、AUCクラスコンストラクタ
        self.miss_class = Miss(
            idx, self.y_pred, y_val, W_val, model, model_folder, test_root, miss_folder)
        self.idx = idx

    def analysis(self):
        # folderごとのtrainingデータの数が分かる
        train_dic = num_count(self.train_root)
        test_dic = num_count(self.test_root)

        # idxのテストデータにおけるloss,正答率が分かる
        loss_model, accuracy_model = self.basic.accuracy()

        # missの数を出力。その際に、回答一覧のｃｓｖファイルも作成
        miss = self.miss_class.miss_csv_making()
        return train_dic, test_dic, loss_model, accuracy_model, miss

    def result_csv_making_0(self):
        '''
        初回に行名のみを記入するメソッド
        '''

        columns_list = []
        train_column_list = [
            "train" + "(" + x + ")" for x in os.listdir(self.train_root)]
        test_column_list = [
            "test" + "(" + x + ")" for x in os.listdir(self.test_root)]

        columns_list.extend(train_column_list)
        columns_list.extend(test_column_list)
        columns_list.extend(["loss", "accuracy", "miss"])
        df = pd.DataFrame(columns=[columns_list])
        df.to_csv(self.result_file, encoding="utf-8", index=False)
        return

    def result_csv_making(self):
        train_dic, test_dic, loss_model, accuracy_model, miss = AnalysisMulti.analysis(
            self)
        df = pd.read_csv(self.result_file)
        df2 = pd.DataFrame()
        for column in train_dic.keys():
            df2["train" + "(" + column + ")"] = [train_dic[column]]
        for column in test_dic.keys():
            df2["test" + "(" + column + ")"] = [test_dic[column]]

        df2["loss"] = [loss_model]
        df2["accuracy"] = [accuracy_model]
        df2["miss"] = [miss]
        df = df.append(df2)
        df.to_csv(self.result_file, encoding="utf-8", index=False)
        return

    def result_csv(self):
        if self.idx == 0:
            AnalysisMulti.result_csv_making_0(self)
        AnalysisMulti.result_csv_making(self)
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


def miss_summarize(miss_folder, k, miss_file):
    df2 = pd.DataFrame()
    for i in range(k):
        csv_file = "miss_" + str(i) + ".csv"
        csv_fpath = os.path.join(miss_folder, csv_file)
        df = pd.read_csv(csv_fpath, encoding="utf-8")
        df2 = df2.append(df)
    df2.to_csv(miss_file, encoding="utf-8", index=False)
    return


def roc_auc_ci(y_true, y_score, alpha, positive=1):
    AUC = roc_auc_score(y_true, y_score)
    # 有病グループの数がN1, 正常群の数がN2
    N1 = sum(y_true == positive)
    N2 = sum(y_true != positive)
    # Q1は
    Q1 = AUC / (2 - AUC)
    Q2 = 2*AUC**2 / (1 + AUC)
    SE_AUC = sqrt((AUC*(1 - AUC) + (N1 - 1)*(Q1 - AUC**2) + (N2 - 1)*(Q2 - AUC**2)) / (N1*N2))
    a,b = stats.norm.interval(alpha, loc=0, scale=1)
    lower = AUC + a*SE_AUC
    upper = AUC + b*SE_AUC
    if lower < 0:
        lower = 0
    if upper > 1:
        upper = 1
    return [AUC,lower, upper]



def summary_analysis(miss_summary_file, summary_file, roc_fig, img_folder,alpha):
    '''
    AnalysisBinaryで作成されたcsvファイルを分析
    '''

    df = pd.read_csv(miss_summary_file, encoding="utf-8")
    df0 = df[df["true"] == 0]
    df1 = df[df["true"] == 1]
    n_normal = len(df0)
    n_des = len(df1)
    y_pred = np.array(df[os.listdir(img_folder)[1]])
    y_true = np.array(df["true"])

    # AUCについて
    # y_pred, y_trueを用いて95%信頼区間を求める
    AUCs = roc_auc_ci(y_true, y_pred, alpha, positive = 1)
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
    plt.scatter(np.array(1-specificity), np.array(sensitivity), s = 50, c = "green")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc="lower right")
    plt.savefig(roc_fig)

    return
