import numpy as np
import cv2
import os


def read_path():

    data_x = []
    data_y = []

    filename = os.listdir('crop_ccpd')
    filename.sort()
    for name in filename:

        img_path = 'crop_ccpd/' + name
        obj1 = name.split('.')
        obj2 = obj1[0]
        obj3 = obj2.split('_')

        x1 = int(obj3[1])
        y1 = int(obj3[2])
        x2 = int(obj3[3])
        y2 = int(obj3[4])

        data_x.append(img_path)
        data_y.append([x1, y1, x2, y2])

    return data_x, data_y


def make_data():

    data_x, data_y = read_path()
    print('all image quantity : ', len(data_y))    # 3116

    train_x = data_x[:3000]
    train_y = data_y[:3000]
    val_x = data_x[3000:]
    val_y = data_y[3000:]
    test_x = data_x[3000:]
    test_y = data_y[3000:]

    return train_x, train_y, val_x, val_y, test_x, test_y

