#! env python
# -*- coding: utf-8 -*-

import os
import sys

# Projects.__init__.py
# Date: 2018/04/01
# Filename: __init__.py 
# To change this template, choose Tools | Templates
# and open the template in the editor.
__author__ = 'masuo'
__date__ = "2018/04/01"


def main():
    # 作業ディレクトリを自身のファイルのディレクトリに変更
    os.chdir(os.path.dirname(os.path.abspath(sys.argv[0])))
    return


if __name__ == '__main__':
    main()