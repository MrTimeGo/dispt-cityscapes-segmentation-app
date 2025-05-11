import os
import numpy as np
import cv2 as cv
from tqdm import tqdm
import tensorflow as tf
from keras.preprocessing.image import img_to_array

ROOT='./..'

X_DIR = '/leftImg8bit_trainvaltest/leftImg8bit'
Y_DIR = '/gtFine_trainvaltest/gtFine'

TRAIN_DIR = '/train'
TEST_DIR = '/val'
VAL_DIR = '/test'

def read_x_train(width, height, count = -1):
    return read_x_and_resize(f'{ROOT}{X_DIR}{TRAIN_DIR}', width, height, count)

def read_y_train(width, height, count = -1):
    return read_y_and_resize(f'{ROOT}{Y_DIR}{TRAIN_DIR}', width, height, count)

def read_x_test(width, height):
    return read_x_and_resize(f'{ROOT}{X_DIR}{TEST_DIR}', width, height)

def read_y_test(width, height):
    return read_y_and_resize(f'{ROOT}{Y_DIR}{TEST_DIR}', width, height)

def read_x_val(width, height, count):
    return read_x_and_resize(f'{ROOT}{X_DIR}{VAL_DIR}', width, height, count)

def read_y_val(width, height, count):
    return read_y_and_resize(f'{ROOT}{Y_DIR}{VAL_DIR}', width, height, count)


def read_x_and_resize(path, width, height, count = -1):
    image_paths = []
    for path, _, files in os.walk(path):
        image_paths.extend(
            map(lambda file: f'{path}/{file}', files)
        )

    image_paths = sorted(image_paths)
    if count > 0:
        image_paths = image_paths[:count]
    return read_imgs(image_paths, width, height)


def read_y_and_resize(path, width, height, count = -1):
    masks_paths = []
    for path, _, files in os.walk(path):
        masks_paths.extend(
            list(map(lambda file: f'{path}/{file}', filter(lambda file: 'color' in file, files)))
        )

    masks_paths = sorted(masks_paths)
    if count > 0:
        masks_paths = masks_paths[:count]

    return read_imgs(masks_paths, width, height)


def read_imgs(paths, width, height):
    images = np.zeros(shape=(len(paths), width, height, 3))
    for i in tqdm(range(len(paths))):
        image_path = paths[i]
        image = cv.imread(image_path)
        #print(image[int(image.shape[0] / 2)][int(image.shape[1] / 2)])
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image = img_to_array(image).astype('float')
        #image = image / 255.0
        #print(image[int(image.shape[0] / 2)][int(image.shape[1] / 2)])
        img = tf.image.resize(image, (width, height), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        #print(image[int(image.shape[0] / 2)][int(image.shape[1] / 2)])

        images[i] = img

    return images