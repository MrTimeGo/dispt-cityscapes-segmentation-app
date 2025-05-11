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
model = create_model_and_compile(SIZE, SIZE, len(labels))

model.load_weights('../cs-checkpoint.h5')
print('weights loaded')

def evaluate(img, skip_labels):
    original_size = img.shape[:2]
    img = cv.resize(img, (SIZE, SIZE), interpolation=cv.INTER_NEAREST)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = img / 255

    x = img[np.newaxis, ...]
    with tf.device('/device:GPU:0'):
        y = model.predict(x)

    mask_rgb = masks_classes_to_rgb(y, skip_labels)
    mask_rgb = mask_rgb[0]
    mask_rgb = np.float32(mask_rgb)
    mask_rgb = cv.cvtColor(mask_rgb, cv.COLOR_RGB2BGR)
    mask_rgb = cv.resize(mask_rgb, (original_size[1], original_size[0]), interpolation=cv.INTER_CUBIC)
    mask_rgb = cv.GaussianBlur(mask_rgb,(5,5),0)
    mask_rgb = cv.imencode('.png', mask_rgb)[1].tobytes()

    return mask_rgb


# X_val= read_x_val(SIZE, SIZE, 1)[0]
#
# X_val = np.float32(X_val)
#
# y_val = evaluate(X_val, ['sidewalk'])
#
# plt.figure()
# plt.subplot(1, 2, 1)
# plt.imshow(X_val / 255)
# plt.subplot(1, 2, 2)
# plt.imshow(y_val / 255)
# plt.show()



#
# with tf.device('/device:GPU:0'):
#     model = create_model_and_compile(SIZE, SIZE, len(labels))
#
#     model.load_weights('../cs-checkpoint.h5')
#
#     X_val= read_x_val(SIZE, SIZE, 5)
#
#     X_val = X_val / 255
#
#     raw_pred_mask = model.predict(X_val)
#
#     pred_mask = masks_classes_to_rgb(raw_pred_mask)
#
#     # plt.figure()
#     # plt.imshow(pred_mask[0])
#     #
#     # plt.show()
#
#
#     if not os.path.isdir('../outputs'):
#         os.makedirs('../outputs')
#
#     X_val = X_val * 255
#
#     pred_mask = np.float32(pred_mask)
#     X_val = np.float32(X_val)
#     for i in range(len(pred_mask)):
#         mask = cv.cvtColor(pred_mask[i], cv.COLOR_RGB2BGR)
#         cv.imwrite(f'../outputs/y_{i}.png', mask)
#
#         image = cv.cvtColor(X_val[i], cv.COLOR_RGB2BGR)
#         cv.imwrite(f'../outputs/x_{i}.png', image)
#
