#!/usr/bin/env python
# -*- coding: utf-8 -*-

# フォルダ操作に関する処理集

import os
import shutil


def folder_create(folder):
    if not os.path.isdir(folder):
        os.mkdir(folder)
    return


def folder_delete(folder):
    if os.path.isdir(folder):
        shutil.rmtree(folder)


def folder_clean(img_root):
    '''
    `img_root`フォルダ中のdesktop.iniの削除
    '''
    folder_list = os.listdir(img_root)
    for folder in folder_list:
        folderpath = os.path.join(img_root, folder)
        if folder == "desktop.ini":
            os.remove(folderpath)
        else:
            file_list = os.listdir(folderpath)
            for file in file_list:
                if file == "desktop.ini":
                    fpath = os.path.join(folderpath, file)
                    os.remove(fpath)
    return
