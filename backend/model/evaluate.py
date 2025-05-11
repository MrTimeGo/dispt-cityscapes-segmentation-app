import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2 as cv
import os
from keras.preprocessing.image import array_to_img

from .model_instances.u_net_a import create_model_and_compile
from .utils.mask_rgb_to_classes import labels, masks_classes_to_rgb
from .utils.image_reader import read_x_val, read_y_val

SIZE = 160

with tf.device('/device:GPU:0'):
    model = create_model_and_compile(SIZE, SIZE, len(labels))

    model.load_weights('../cs-checkpoint.h5')

    X_val= read_x_val(SIZE, SIZE, 5)

    X_val = X_val / 255

    raw_pred_mask = model.predict(X_val)

    pred_mask = masks_classes_to_rgb(raw_pred_mask)

    # plt.figure()
    # plt.imshow(pred_mask[0])
    #
    # plt.show()


    if not os.path.isdir('../outputs'):
        os.makedirs('../outputs')

    X_val = X_val * 255

    pred_mask = np.float32(pred_mask)
    X_val = np.float32(X_val)
    for i in range(len(pred_mask)):
        mask = cv.cvtColor(pred_mask[i], cv.COLOR_RGB2BGR)
        cv.imwrite(f'../outputs/y_{i}.png', mask)

        image = cv.cvtColor(X_val[i], cv.COLOR_RGB2BGR)
        cv.imwrite(f'../outputs/x_{i}.png', image)

