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
