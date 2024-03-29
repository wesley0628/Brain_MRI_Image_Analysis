from keras import layers, models, regularizers
from keras.applications.vgg19 import *
from keras.applications.resnet import *
from keras.applications.efficientnet import *
from keras.applications.vgg16 import *
from keras.layers import GlobalAveragePooling2D, BatchNormalization


def simple_cnn():
    model_simple_cnn = models.Sequential()
    model_simple_cnn.add(layers.Conv2D(32, 3, activation='relu', input_shape=(224, 224, 3)))
    model_simple_cnn.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model_simple_cnn.add(layers.Conv2D(32, 3, activation='relu'))
    model_simple_cnn.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model_simple_cnn.add(layers.Conv2D(64, 3, activation='relu'))
    model_simple_cnn.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model_simple_cnn.add(layers.Flatten())
    model_simple_cnn.add(layers.Dense(256, activation='softmax'))
    model_simple_cnn.add(layers.Dropout(0.5))
    model_simple_cnn.add(layers.Dense(4, activation='softmax'))
    return model_simple_cnn


def resnet_50():
    base_resnet50 = ResNet50()
    model_resnet = models.Sequential()
    model_resnet.add(base_resnet50)
    model_resnet.add(layers.Input(shape=(224, 224, 3)))
    model_resnet.add(layers.Flatten())
    model_resnet.add(layers.Dense(256, activation='relu'))
    model_resnet.add(layers.Dropout(0.5))
    model_resnet.add(layers.Dense(4, activation='softmax'))
    return model_resnet


def vgg_16():
    base_vgg16 = VGG16(input_shape=(224, 224, 3))
    model_vgg16 = models.Sequential()
    model_vgg16.add(base_vgg16)
    model_vgg16.add(layers.Flatten())
    model_vgg16.add(layers.Dense(128, activation='relu'))
    model_vgg16.add(layers.Dropout(0.5))
    model_vgg16.add(layers.Dense(4, activation='softmax'))
    return model_vgg16


def vgg_19():
    base_vgg19 = VGG19(input_shape=(224, 224, 3))
    model_vgg19 = models.Sequential()
    model_vgg19.add(base_vgg19)
    model_vgg19.add(layers.Flatten())
    model_vgg19.add(layers.Dense(128, kernel_regularizer='l2', activation='relu'))
    model_vgg19.add(layers.Dropout(0.5))
    model_vgg19.add(layers.Dense(4, activation='softmax'))
    return model_vgg19


def efficientnet(base_model):
    base_efficientnet = base_model
    model_efficientnet = models.Sequential()
    model_efficientnet.add(base_efficientnet)
    model_efficientnet.add(layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001))
    model_efficientnet.add(layers.Dense(256, kernel_regularizer=regularizers.l2(l=0.016),
                                        activity_regularizer=regularizers.l1(0.006),
                           bias_regularizer=regularizers.l1(0.006), activation='relu'))
    model_efficientnet.add(layers.Dropout(0.5))
    model_efficientnet.add(layers.Dense(4, activation='softmax'))
    return model_efficientnet

