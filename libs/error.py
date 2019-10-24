#!/usr/bin/env python
# -*- coding: utf-8 -*-

# エラー出力に関する処理
# 100番台 ファイルやフォルダが存在しない時
# 200番台 オプションファイルの設定が間違っている時

import sys

PREFIX = "--- Error --------------------------------------------------"
SUFFIX = "------------------------------------------------------------"


def option_file_not_exist():
    '''
    options.confファイルが存在しなかったときに発生
    終了コード : 100
    '''
    print(PREFIX)
    print("Error Code 100: options.conf is not found.")
    print("put options.conf in root directory")
    print(SUFFIX)

    sys.exit(100)
    return


def section_not_found(section_name):
    '''
    options.conf内に規定のセクションが存在しなかったときに発生
    終了コード : 200
    '''
    print(PREFIX)
    print(f"Error Code 200: Section [{section_name}] is not found.")
    print(f"Add [{section_name}] in options.conf.")
    print(SUFFIX)

    sys.exit(200)
    return


def option_not_found(section_name, option_name):
    '''
    options.conf内に規定のオプションが存在しなかったときに発生
    終了コード : 201
    '''
    print(PREFIX)
    print(f"Error Code 201: Option [{option_name}] is not found.")
    print(f"Add [{option_name}] in section [{section_name}].")
    print(SUFFIX)

    sys.exit(201)
    return


def positive_label_not_found(positive_label):
    '''
    データセットに陽性ラベルとして指定したラベルが無かったときに発生
    終了コード : 300
    '''
    print(PREFIX)
    print(f"Error Code 301: positive label [{positive_label}] is not found.")
    print(f"Check positive label [{positive_label}] is correct.")
    print(SUFFIX)

    sys.exit(300)
    return


def tf_version_error(tf_version):
    '''
    TensorFlowのバージョンが2.x系以上でなかったときに発生
    終了コード : 400
    '''
    print(PREFIX)
    print("Error Code 400: ", end="")
    print(f"The tensorflow version [{tf_version}] is inappropriate.")
    print("Upgrade the tensorflow version to 2.x.")
    print(SUFFIX)

    sys.exit(400)
    return