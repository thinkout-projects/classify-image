#!/usr/bin/env python
# -*- coding: utf-8 -*-

# モデル構築に使用するレイヤー
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Activation
from tensorflow.keras.layers import Conv2D, BatchNormalization
from tensorflow.keras.layers import MaxPooling2D, GlobalAveragePooling2D

# 構築済みネットワーク
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.densenet import DenseNet201, DenseNet169
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.applications.resnet50 import ResNet50



class Models(object):
    '''
    modelを定義するクラス
    '''

    def __init__(self, size, classes, pic_mode):
        self.ch = 3
        self.size = size
        self.w = self.size[0]
        self.h = self.size[1]
        self.classes = classes
        self.pic_mode = pic_mode


    def choose(self, model_name):
        if model_name == 'VGG16':
            model = self.vgg16()
        elif model_name == 'VGG19':
            model = self.vgg19()
        elif model_name == 'DenseNet121':
            model = self.dense121()
        elif model_name == 'DenseNet169':
            model = self.dense169()
        elif model_name == 'DenseNet201':
            model = self.dense201()
        elif model_name == 'InceptionResNetV2':
            model = self.inception_resnet2()
        elif model_name == 'InceptionV3':
            model = self.inception3()
        elif model_name == 'ResNet50':
            model = self.resnet50()
        elif model_name == 'Xception':
            model = self.xception()
        elif model_name == 'LightWeight':
            model = self.light_weight_model()
        return model


    def vgg16(self):
        '''
        VGG16(初期値Imagenet)
        '''

        base_model = VGG16(include_top=False, weights='imagenet',
                           input_shape=(self.h, self.w, self.ch))
        x = GlobalAveragePooling2D()(base_model.output)
        x = Dense(128, kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        if self.classes == 1:
            outputs = Dense(self.classes, kernel_initializer='he_normal',
                            activation='relu')(x)
        else:
            outputs = Dense(self.classes, kernel_initializer='he_normal',
                            activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=outputs)

        return model


    def vgg19(self):
        '''
        VGG19(初期値Imagenet)
        '''

        base_model = VGG19(include_top=False, weights='imagenet',
                           input_shape=(self.h, self.w, self.ch))
        x = GlobalAveragePooling2D()(base_model.output)
        x = Dense(128, kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        if self.classes == 1:
            outputs = Dense(self.classes, kernel_initializer='he_normal',
                            activation='relu')(x)
        else:
            outputs = Dense(self.classes, kernel_initializer='he_normal',
                            activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=outputs)

        return model


    def dense121(self):
        '''
        DenseNet121(初期値Imagenet)
        '''

        base_model = DenseNet121(include_top=False, weights='imagenet',
                                 input_shape=(self.h, self.w, self.ch))
        x = GlobalAveragePooling2D()(base_model.output)
        x = Dense(256, kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        if self.classes == 1:
            outputs = Dense(self.classes, kernel_initializer='he_normal',
                            activation='relu')(x)
        else:
            outputs = Dense(self.classes, kernel_initializer='he_normal',
                            activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=outputs)

        return model


    def dense169(self):
        '''
        DenseNet169(初期値Imagenet)
        '''

        base_model = DenseNet169(include_top=False, weights='imagenet',
                                 input_shape=(self.h, self.w, self.ch))
        x = GlobalAveragePooling2D()(base_model.output)
        x = Dense(512, kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        if self.classes == 1:
            outputs = Dense(self.classes, kernel_initializer='he_normal',
                            activation='relu')(x)
        else:
            outputs = Dense(self.classes, kernel_initializer='he_normal',
                            activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=outputs)

        return model


    def dense201(self):
        '''
        DenseNet201(初期値Imagenet)
        '''

        base_model = DenseNet201(include_top=False, weights='imagenet',
                                 input_shape=(self.h, self.w, self.ch))
        x = GlobalAveragePooling2D()(base_model.output)
        x = Dense(512, kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        if self.classes == 1:
            outputs = Dense(self.classes, kernel_initializer='he_normal',
                            activation='relu')(x)
        else:
            outputs = Dense(self.classes, kernel_initializer='he_normal',
                            activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=outputs)

        return model


    def inception_resnet2(self):
        '''
        InceptionResNetV2(初期値Imagenet)
        '''

        base_model = InceptionResNetV2(include_top=False, weights='imagenet',
                                       input_shape=(self.h, self.w, self.ch))
        x = GlobalAveragePooling2D()(base_model.output)
        x = Dense(512, kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        if self.classes == 1:
            outputs = Dense(self.classes, kernel_initializer='he_normal',
                            activation='relu')(x)
        else:
            outputs = Dense(self.classes, kernel_initializer='he_normal',
                            activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=outputs)

        return model


    def inception3(self):
        '''
        InceptionV3(初期値Imagenet)
        '''

        base_model = InceptionV3(include_top=False, weights='imagenet',
                                 input_shape=(self.h, self.w, self.ch))
        x = GlobalAveragePooling2D()(base_model.output)
        x = Dense(512, kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        if self.classes == 1:
            outputs = Dense(self.classes, kernel_initializer='he_normal',
                            activation='relu')(x)
        else:
            outputs = Dense(self.classes, kernel_initializer='he_normal',
                            activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=outputs)

        return model


    def resnet50(self):
        '''
        ResNet50(初期値Imagenet)
        '''

        base_model = ResNet50(include_top=False, weights='imagenet',
                              input_shape=(self.h, self.w, self.ch))
        x = GlobalAveragePooling2D()(base_model.output)
        x = Dense(512, kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        if self.classes == 1:
            outputs = Dense(self.classes, kernel_initializer='he_normal',
                            activation='relu')(x)
        else:
            outputs = Dense(self.classes, kernel_initializer='he_normal',
                            activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=outputs)

        return model


    def xception(self):
        '''
        Xception(初期値Imagenet)
        '''

        base_model = Xception(include_top=False, weights='imagenet',
                              input_shape=(self.h, self.w, self.ch))
        x = GlobalAveragePooling2D()(base_model.output)
        x = Dense(512, kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        if self.classes == 1:
            outputs = Dense(self.classes, kernel_initializer='he_normal',
                            activation='relu')(x)
        else:
            outputs = Dense(self.classes, kernel_initializer='he_normal',
                            activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=outputs)

        return model


    def light_weight_model(self):
        '''
        laptopでも使用できる3層畳み込みネットワーク
        '''

        inputs = Input(shape=(self.h, self.w, self.ch))
        x = Conv2D(32, (3, 3), padding='same',
                   kernel_initializer='he_normal')(inputs)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Activation('relu')(x)
        x = Conv2D(64, (3, 3), padding='same',
                   kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Activation('relu')(x)
        x = Conv2D(128, (3, 3), padding='same',
                   kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Activation('relu')(x)
        x = GlobalAveragePooling2D()(x)
        if self.classes == 1:
            outputs = Dense(self.classes, kernel_initializer='he_normal',
                            activation='relu')(x)
        else:
            outputs = Dense(self.classes, kernel_initializer='he_normal',
                            activation='softmax')(x)
        model = Model(inputs=inputs, outputs=outputs)

        return model
