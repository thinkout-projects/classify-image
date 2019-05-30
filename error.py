#!/usr/bin/env python
# -*- coding: utf-8 -*-

# エラー出力
#
import sys

PREFIX = "--- Error --------------------------------------------------"
SUFFIX = "------------------------------------------------------------"


def section_not_found(section_name):
    '''
    options.conf内に規定のセクションが存在しなかったときに発生
    終了コード : 100
    '''
    print(PREFIX)
    print(f"Section [{section_name}] is not found.")
    print(f"Add [{section_name}] in options.conf.")
    print(SUFFIX)

    sys.exit(100)
    return


def option_not_found(section_name, option_name):
    '''
    options.conf内に規定のオプションが存在しなかったときに発生
    終了コード : 101
    '''
    print(PREFIX)
    print(f"Option [{option_name}] is not found.")
    print(f"Add [{option_name}] in section [{section_name}].")
    print(SUFFIX)

    sys.exit(101)
    return
