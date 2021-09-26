import numpy as np
import cv2
import os
from read_data_path import make_data
from psp_model import get_psp_model
from train import SequenceData
from train import train_network
from PIL import Image


inputs_size = (473, 473, 3)
n_classes = 2


def detect_box(test_x):

    psp_model = get_psp_model()
    psp_model.load_weights('best_val0.02110.h5')

    # img：得到边框位置
    # semantic ：rgb图像语义分割结果

    for i in range(len(test_x)):

        img = cv2.imread(test_x[i])
        size = img.shape    # (520, 660, 3)

        img1 = cv2.resize(img, (inputs_size[1], inputs_size[0]), interpolation=cv2.INTER_AREA)
        img2 = img1 / 255
        img3 = img2[np.newaxis, :, :, :]

        result1 = psp_model.predict(img3)  # (1, 473, 473, 2)
        result2 = result1[0]
        result3 = cv2.resize(result2, (size[1], size[0]), interpolation=cv2.INTER_NEAREST)    # (520, 660, 2)

        mask = np.zeros((size[0], size[1]))
        semantic = np.zeros((size[0], size[1], 3))

        candidate = []
        for j in range(size[0]):
            for k in range(size[1]):

                index = np.argmax(result3[j, k, :])
                mask[j, k] = index
                if index == 1:
                    candidate.append([j, k])
                    semantic[j, k, :] = np.array([0, 0, 255])    # 浅蓝色
                else:
                    semantic[j, k, :] = np.array([255, 255, 0])  # 橘红色

        candidate = np.array(candidate)

        if len(candidate) > 0:

            a1 = int(np.min(candidate[:, 1]))
            b1 = int(np.min(candidate[:, 0]))
            a2 = int(np.max(candidate[:, 1]))
            b2 = int(np.max(candidate[:, 0]))

            cv2.rectangle(img, (a1, b1), (a2, b2), (0, 0, 255), 2)

        # cv2.namedWindow("img")
        # cv2.imshow("img", img)
        # cv2.waitKey(0)
        #
        # cv2.namedWindow("semantic")
        # cv2.imshow("semantic", semantic)
        # cv2.waitKey(0)

        cv2.imwrite("demo/" + str(i) + '_semantic' + '.jpg', semantic)
        cv2.imwrite("demo/" + str(i) + '_img' + '.jpg', img/1.0)





