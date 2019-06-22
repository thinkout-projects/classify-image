#!/usr/bin/env python
# -*- coding: utf-8 -*-

# フォルダ操作に関する処理

import os
import shutil


def folder_create(folder):
    if not os.path.isdir(folder):
        os.mkdir(folder)
    return


def folder_delete(folder):
    if os.path.isdir(folder):
        shutil.rmtree(folder)
    return
