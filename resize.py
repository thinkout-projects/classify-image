#! env python
# -*- coding: utf-8 -*-

import os
import sys
import cv2

# Projects.__init__.py
# Date: 2018/04/01
# Filename: __init__.py
# To change this template, choose Tools | Templates
# and open the template in the editor.
__author__ = 'masuo'
__date__ = "2018/04/01"
def fol_make(folder):
    if not os.path.isdir(folder):
        os.mkdir(folder)


def resizing():
    # 作業ディレクトリを自身のファイルのディレクトリに変更
    root1 = "img_old"
    root2 = "img"
    fol_make(root2)
    folder_list = os.listdir(root1)
    for folder in folder_list:
        folpath1 = os.path.join(root1, folder)
        folpath2 = os.path.join(root2, folder)
        fol_make(folpath2)
        file_list = os.listdir(folpath1)
        for file in file_list:
            fpath = os.path.join(folpath1, file)
            newfpath = os.path.join(folpath2, file)
            img = cv2.imread(fpath)
            newimg = cv2.resize(img,(256,192))
            cv2.imwrite(newfpath, newimg)

if __name__ == '__main__':
    main()
