import numpy as np
import cv2
import os
from read_data_path import make_data
from psp_model import get_psp_model
from train import SequenceData
from train import train_network
from train import load_network_then_train
from detect import detect_box

# 调用GPU进行训练
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


if __name__ == "__main__":

    train_x, train_y, val_x, val_y, test_x, test_y = make_data()

    psp_model = get_psp_model()
    psp_model.summary()

    train_generator = SequenceData(train_x[:32], train_y[:32], 16)
    test_generator = SequenceData(test_x, test_y, 16)

    # train_network(train_generator, test_generator, epoch=20)
    # load_network_then_train(train_generator, test_generator, epoch=20,
    #                         input_name='first_weights.hdf5',
    #                         output_name='second_weights.hdf5')

    detect_box(test_x)








