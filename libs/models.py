# VGG16のネットワーク系
from keras.models import Sequential, Model

from keras.layers import Input, Flatten, Dense, Dropout, Activation
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.layers.convolutional import Conv2D, MaxPooling2D

# VGG 16
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.xception import Xception
from keras.applications.inception_v3 import InceptionV3
from keras.applications.densenet import DenseNet201, DenseNet169, DenseNet121
from keras.applications.resnet50 import ResNet50

class Models(object):
    '''
    modelを定義するクラス
    '''
    # InceptionResNetV2
    # Xception
    # InceptionV3
    # DenseNet201
    # ResNet50

    def __init__(self, size, classes, pic_mode):
        self.ch = 3
        self.size = size
        self.w = self.size[0]
        self.h = self.size[1]
        self.classes = classes
        self.pic_mode = pic_mode

    def inception_resnet2(self):
        input_tensor = Input(shape=(self.h, self.w, self.ch))
        design_model = InceptionResNetV2(
            include_top=False, weights='imagenet', input_tensor=input_tensor)

        # FC層を構築
        top_model = Sequential()
        top_model.add(Flatten(input_shape=design_model.output_shape[1:]))
        top_model.add(Dense(256, activation='relu'))
        top_model.add(Dropout(0.5))
        if(self.pic_mode != 2):
            top_model.add(Dense(self.classes, activation='softmax'))
        else:
            top_model.add(Dense(self.classes, activation='relu'))

        # VGG16とFCを接続
        model = Model(input=design_model.input,
                      output=top_model(design_model.output))
        return model

    def xception(self):
        input_tensor = Input(shape=(self.h, self.w, self.ch))
        design_model = Xception(
            include_top=False, weights='imagenet', input_tensor=input_tensor)

        # FC層を構築
        top_model = Sequential()
        top_model.add(Flatten(input_shape=design_model.output_shape[1:]))
        top_model.add(Dense(256, activation='relu'))
        top_model.add(Dropout(0.5))
        if(self.pic_mode != 2):
            top_model.add(Dense(self.classes, activation='softmax'))
        else:
            top_model.add(Dense(self.classes, activation='relu'))

        # VGG16とFCを接続
        model = Model(input=design_model.input,
                      output=top_model(design_model.output))
        return model

    def inception3(self):
        input_tensor = Input(shape=(self.h, self.w, self.ch))
        design_model = InceptionV3(
            include_top=False, weights='imagenet', input_tensor=input_tensor)

        # FC層を構築
        top_model = Sequential()
        top_model.add(Flatten(input_shape=design_model.output_shape[1:]))
        top_model.add(Dense(256, activation='relu'))
        top_model.add(Dropout(0.5))
        if(self.pic_mode != 2):
            top_model.add(Dense(self.classes, activation='softmax'))
        else:
            top_model.add(Dense(self.classes, activation='relu'))

        # VGG16とFCを接続
        model = Model(input=design_model.input,
                      output=top_model(design_model.output))
        return model

    def dense121(self):
        input_tensor = Input(shape=(self.h, self.w, self.ch))
        design_model = DenseNet121(
            include_top=False, weights='imagenet', input_tensor=input_tensor)

        # FC層を構築
        top_model = Sequential()
        top_model.add(Flatten(input_shape=design_model.output_shape[1:]))
        top_model.add(Dense(256, activation='relu'))
        top_model.add(Dropout(0.5))
        if(self.pic_mode != 2):
            top_model.add(Dense(self.classes, activation='softmax'))
        else:
            top_model.add(Dense(self.classes, activation='relu'))

        # VGG16とFCを接続
        model = Model(input=design_model.input,
                      output=top_model(design_model.output))
        return model

    def dense169(self):
        input_tensor = Input(shape=(self.h, self.w, self.ch))
        design_model = DenseNet169(
            include_top=False, weights='imagenet', input_tensor=input_tensor)

        # FC層を構築
        top_model = Sequential()
        top_model.add(Flatten(input_shape=design_model.output_shape[1:]))
        top_model.add(Dense(256, activation='relu'))
        top_model.add(Dropout(0.5))
        if(self.pic_mode != 2):
            top_model.add(Dense(self.classes, activation='softmax'))
        else:
            top_model.add(Dense(self.classes, activation='relu'))

        # VGG16とFCを接続
        model = Model(input=design_model.input,
                      output=top_model(design_model.output))
        return model

    def dense201(self):
        input_tensor = Input(shape=(self.h, self.w, self.ch))
        design_model = DenseNet201(
            include_top=False, weights='imagenet', input_tensor=input_tensor)

        # FC層を構築
        top_model = Sequential()
        top_model.add(Flatten(input_shape=design_model.output_shape[1:]))
        top_model.add(Dense(256, activation='relu'))
        top_model.add(Dropout(0.5))
        if(self.pic_mode != 2):
            top_model.add(Dense(self.classes, activation='softmax'))
        else:
            top_model.add(Dense(self.classes, activation='relu'))

        # VGG16とFCを接続
        model = Model(input=design_model.input,
                      output=top_model(design_model.output))
        return model

    def resnet50(self):
        input_tensor = Input(shape=(self.h, self.w, self.ch))
        design_model = ResNet50(
            include_top=False, weights='imagenet', input_tensor=input_tensor)

        # FC層を構築
        top_model = Sequential()
        top_model.add(Flatten(input_shape=design_model.output_shape[1:]))
        top_model.add(Dense(256, activation='relu'))
        top_model.add(Dropout(0.5))
        if(self.pic_mode != 2):
            top_model.add(Dense(self.classes, activation='softmax'))
        else:
            top_model.add(Dense(self.classes, activation='relu'))

        # VGG16とFCを接続
        model = Model(input=design_model.input,
                      output=top_model(design_model.output))
        return model

    def vgg19(self):
        input_tensor = Input(shape=(self.h, self.w, self.ch))
        design_model = VGG19(
            include_top=False, weights='imagenet', input_tensor=input_tensor)

        # FC層を構築
        top_model = Sequential()
        top_model.add(Flatten(input_shape=design_model.output_shape[1:]))
        top_model.add(Dense(256, activation='relu'))
        top_model.add(Dropout(0.5))
        if(self.pic_mode != 2):
            top_model.add(Dense(self.classes, activation='softmax'))
        else:
            top_model.add(Dense(self.classes, activation='relu'))

        # VGG16とFCを接続
        model = Model(input=design_model.input,
                      output=top_model(design_model.output))
        return model

    def vgg16(self):
        '''
        VGG16(初期値Imagenet、非固定版)
        '''

        input_tensor = Input(shape=(self.h, self.w, self.ch))
        vgg16_model = VGG16(include_top=False,
                            weights='imagenet', input_tensor=input_tensor)

        # FC層を構築
        top_model = Sequential()
        top_model.add(Flatten(input_shape=vgg16_model.output_shape[1:]))
        top_model.add(Dense(256, activation='relu'))
        top_model.add(Dropout(0.5))
        if(self.pic_mode != 2):
            top_model.add(Dense(self.classes, activation='softmax'))
        else:
            top_model.add(Dense(self.classes, activation='relu'))

        # VGG16とFCを接続
        model = Model(input=vgg16_model.input,
                      output=top_model(vgg16_model.output))
        return model

    def test_model(self):
        '''
        laptopでも使用できる3層ネットワーク
        inputなどが正しいか評価するときに使用
        '''

        model = Sequential()
        model.add(Conv2D(32, (3, 3), padding='valid',
                         input_shape=(self.h, self.w, self.ch)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, (3, 3), padding='valid'))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (3, 3), padding='valid'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.classes))

        if(self.pic_mode != 2):
            model.add(Activation('softmax'))
        else:
            model.add(Activation('relu'))
        return model
