import cv2
import os
import random
import numpy as np
from keras.utils import Sequence
import math
from psp_model import get_psp_model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from PIL import Image
from keras import optimizers


inputs_size = (473, 473, 3)
n_classes = 1 + 1


class SequenceData(Sequence):

    def __init__(self, data_x, data_y, batch_size):
        self.batch_size = batch_size
        self.data_x = data_x
        self.data_y = data_y
        self.indexes = np.arange(len(self.data_x))

    def __len__(self):
        return math.floor(len(self.data_x) / float(self.batch_size))

    def on_epoch_end(self):
        np.random.shuffle(self.indexes)

    def __getitem__(self, idx):

        batch_index = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = [self.data_x[k] for k in batch_index]
        batch_y = [self.data_y[k] for k in batch_index]

        x = np.zeros((self.batch_size, inputs_size[1], inputs_size[0], 3))
        y = np.zeros((self.batch_size, inputs_size[1], inputs_size[0], n_classes))

        for i in range(self.batch_size):

            img = cv2.imread(batch_x[i])
            size = img.shape
            img1 = cv2.resize(img, (inputs_size[1], inputs_size[0]), interpolation=cv2.INTER_AREA)
            img2 = img1 / 255
            x[i, :, :, :] = img2

            label = np.zeros((inputs_size[1], inputs_size[0])).astype('int64')

            x1 = int(batch_y[i][0] / size[1] * inputs_size[1])
            y1 = int(batch_y[i][1] / size[0] * inputs_size[0])
            x2 = int(batch_y[i][2] / size[1] * inputs_size[1])
            y2 = int(batch_y[i][3] / size[0] * inputs_size[0])

            label[y1:y2, x1:x2] = 1

            # 调用eye函数和reshape函数，很方便将数据结构转化为语义分割形式
            label1 = np.eye(n_classes)[label.reshape([-1])]
            label2 = label1.reshape((inputs_size[1], inputs_size[0], n_classes))

            y[i, :, :, :] = label2

            # 用来测试读取的label是否会出错的，demon记录该图像上所有类别的种类
            # demon = []
            # for i1 in range(label.shape[0]):
            #     for j1 in range(label.shape[1]):
            #         demon.append(label[i1, j1])
            # print(set(demon))

            # cv2.namedWindow("Image")
            # cv2.imshow("Image", img2)
            # cv2.waitKey(0)

            # cv2.namedWindow("seg1")
            # cv2.imshow("seg1", label/1.0)
            # cv2.waitKey(0)

        return x, y


def train_network(train_generator, validation_generator, epoch):

    model = get_psp_model()

    # 迁移学习，加载网上下载的部分权重
    model.load_weights('download_weights.h5', by_name=True, skip_mismatch=True)
    print('PSPNet网络层总数为：', len(model.layers))    # 175

    freeze_layers = 146
    for i in range(freeze_layers):
        model.layers[i].trainable = False
        print(model.layers[i].name)

    adam = Adam(lr=1e-4)
    log_dir = "Logs/1/"
    checkpoint = ModelCheckpoint(log_dir + 'epoch{epoch:03d}_val{val_loss:.5f}.h5',
                                 monitor='val_loss', save_weights_only=True, save_best_only=False, period=1)

    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    model.fit_generator(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=epoch,
        validation_data=validation_generator,
        validation_steps=len(validation_generator),
        callbacks=[checkpoint]
    )

    model.save_weights('first_weights.hdf5')


def load_network_then_train(train_generator, validation_generator, epoch, input_name, output_name):

    model = get_psp_model()
    model.load_weights(input_name)
    print('PSPNet网络层总数为：', len(model.layers))  # 175

    freeze_layers = 146
    for i in range(freeze_layers):
        model.layers[i].trainable = False
        print(model.layers[i].name)

    adam = Adam(lr=1e-4)
    sgd = optimizers.SGD(lr=1e-4, momentum=0.9)
    log_dir = "Logs/2/"
    checkpoint = ModelCheckpoint(log_dir + 'epoch{epoch:03d}_val{val_loss:.5f}.h5',
                                 monitor='val_loss', save_weights_only=True, save_best_only=False, period=1)

    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    model.fit_generator(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=epoch,
        validation_data=validation_generator,
        validation_steps=len(validation_generator),
        callbacks=[checkpoint]
    )

    model.save_weights(output_name)
