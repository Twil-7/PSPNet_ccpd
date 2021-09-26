import tensorflow as tf
import numpy as np

from keras.initializers import random_normal
from keras.layers import *
from keras.models import *
from keras import backend as K

from mobile_netv2 import get_mobilenet_encoder


n_classes = 1 + 1


def resize_images(args):

    x = args[0]
    y = args[1]

    return tf.image.resize(x, (K.int_shape(y)[1], K.int_shape(y)[2]))


def pool_block(feats, pool_factor, out_channel):

    h = K.int_shape(feats)[1]    # 30
    w = K.int_shape(feats)[2]    # 30

    #   分区域进行平均池化
    #   strides   = [30,30], [15,15], [10,10], [5, 5]
    #   pool size = 30/1=30  30/2=15  30/3=10  30/6=5

    pool_size = [int(np.round(float(h)/pool_factor)), int(np.round(float(w)/pool_factor))]
    strides = pool_size
    # [30,30] or [15,15] or [10,10] or [5, 5]

    x = AveragePooling2D(pool_size, strides=strides, padding='same')(feats)    # (None, 1, 1, 80)

    #   利用1x1卷积进行通道数的调整
    x = Conv2D(out_channel//4, (1, 1), kernel_initializer=random_normal(stddev=0.02),
               padding='same', use_bias=False)(x)                              # (None, 1, 1, 80)
    x = BatchNormalization()(x)                                                # (None, 1, 1, 80)
    x = Activation('relu')(x)                                                  # (None, 1, 1, 80)

    #   利用resize扩大特征层面, 将(1, 1), (2, 2), (3, 3), (6, 6)上采样恢复到(30, 30)
    x = Lambda(resize_images)([x, feats])                                      # (None, 30, 30, 80)

    return x


def get_psp_model():

    #   通过mobile_net特征提取， 获得两个特征层:  f4为辅助分支 - (None, 30, 30, 96)  ;  o为主干部分 - (None, 30, 30, 320)

    img_input, f4, o = get_mobilenet_encoder()
    out_channel = 320

    # PSP模块，分区域进行池化，将30*30的feature map，分别池化成1x1的区域，2x2的区域，3x3的区域，6x6的区域

    # pool_outs列表 ：
    # [主干部分 - (None, 30, 30, 320),
    # 由1*1扩展而成的 - (None, 30, 30, 80),
    # 由2*2扩展而成的 - (None, 30, 30, 80),
    # 由3*3扩展而成的 - (None, 30, 30, 80),
    # 由6*6扩展而成的 - (None, 30, 30, 80)]

    pool_factors = [1, 2, 3, 6]
    pool_outs = [o]
    for p in pool_factors:
        pooled = pool_block(o, p, out_channel)
        pool_outs.append(pooled)

    #   将获取到的特征层进行堆叠
    #   (30, 30, 320) + (30, 30, 80) + (30, 30, 80) + (30, 30, 80) + (30, 30, 80) = (30, 30, 640)

    o = Concatenate(axis=-1)(pool_outs)

    # 30, 30, 640 -> 30, 30, 80
    o = Conv2D(out_channel//4, (3, 3), kernel_initializer=random_normal(stddev=0.02),
               padding='same', use_bias=False)(o)    # (None, 30, 30, 80)
    o = BatchNormalization()(o)                      # (None, 30, 30, 80)
    o = Activation('relu')(o)                        # (None, 30, 30, 80)

    # 防止过拟合
    o = Dropout(0.1)(o)                              # (None, 30, 30, 80)

    # 利用特征获得预测结果
    # 30, 30, 80 -> 30, 30, 2 -> 473, 473, 2

    o = Conv2D(n_classes, (1, 1), kernel_initializer=random_normal(stddev=0.02),
               padding='same')(o)                    # (None, 30, 30, 2)
    o = Lambda(resize_images)([o, img_input])        # (None, 473, 473, 2)

    #   获得每一个像素点属于每一个类的概率
    o = Activation("softmax", name="main")(o)        # (None, 473, 473, 2)

    model = Model(img_input, o)
    return model
