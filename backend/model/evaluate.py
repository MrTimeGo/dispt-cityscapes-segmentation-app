import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2 as cv
import os

from .model_instances.u_net_a import create_model_and_compile
from .utils.mask_rgb_to_classes import labels, masks_classes_to_rgb
from .utils.image_reader import read_x_val, read_y_val

SIZE = 128

with tf.device('/device:GPU:0'):
    model = create_model_and_compile(SIZE, SIZE, len(labels))

    model.load_weights('../cs-checkpoint.h5')

    X_val, Y_val = read_x_val(SIZE, SIZE, 5), read_y_val(SIZE, SIZE, 5)

    raw_pred_mask = model.predict(X_val)

    pred_mask = masks_classes_to_rgb(raw_pred_mask)

    plt.figure()
    plt.imshow(pred_mask[0])

    plt.show()


    # if not os.path.isdir('../outputs'):
    #     os.makedirs('../outputs')
    #
    #
    # for i in range(len(pred_mask)):
    #     cv.imwrite(f'../outputs/{i}.png', pred_mask[i])

