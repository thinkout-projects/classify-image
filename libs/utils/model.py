#!/usr/bin/env python
# -*- coding: utf-8 -*-

# モデル操作に関する処理

import os
import glob


def model_compile(model, loss, optimizer):
    model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])
    return


def model_selector(model_name, model_ch):
    if model_name == 'VGG16':
        model = model_ch.vgg16()
    elif model_name == 'VGG19':
        model = model_ch.vgg19()
    elif model_name == 'DenseNet121':
        model = model_ch.dense121()
    elif model_name == 'DenseNet169':
        model = model_ch.dense169()
    elif model_name == 'DenseNet201':
        model = model_ch.dense201()
    elif model_name == 'InceptionResNetV2':
        model = model_ch.inception_resnet2()
    elif model_name == 'InceptionV3':
        model = model_ch.inception3()
    elif model_name == 'ResNet50':
        model = model_ch.resnet50()
    elif model_name == 'Xception':
        model = model_ch.xception()

    return model

def model_load(model, model_folder, idx):
    model_files = glob.glob(os.path.join(
        model_folder, "weights_" + str(idx) + "_*"))
    model_files = sorted(model_files)
    model_fpath = model_files[-1]
    model.load_weights(model_fpath)
    return


def model_delete(model, model_folder, idx):
    model_files = glob.glob(os.path.join(
        model_folder, "weights_" + str(idx) + "_*"))
    model_files = sorted(model_files)
    for model_fpath in model_files[:-1]:
        os.remove(model_fpath)
    return
