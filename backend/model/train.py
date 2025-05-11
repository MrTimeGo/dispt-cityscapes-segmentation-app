import pickle

import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

from keras.callbacks import EarlyStopping, ModelCheckpoint

from .model_instances.show_progress import ShowProgress
from .utils.image_reader import read_x_train, read_y_train, read_x_test, read_y_test
from .utils.mask_rgb_to_classes import masks_rgb_to_classes, labels, masks_classes_to_rgb
from .model_instances.u_net_a import create_model_and_compile

SIZE = 160
DUMP_FOLDER_NAME = '../image-loaded-checkpoints'

if os.path.exists(f'{DUMP_FOLDER_NAME}/X_train.pkl'):
    with open(f'{DUMP_FOLDER_NAME}/X_train.pkl', 'rb') as f:
        X_train = pickle.load(f)
    with open(f'{DUMP_FOLDER_NAME}/Y_train.pkl', 'rb') as f:
        Y_train = pickle.load(f)
    with open(f'{DUMP_FOLDER_NAME}/X_test.pkl', 'rb') as f:
        X_test= pickle.load(f)
    with open(f'{DUMP_FOLDER_NAME}/Y_test.pkl', 'rb') as f:
        Y_test = pickle.load(f)
    print('loaded')
else:
    X_train, Y_train = read_x_train(SIZE, SIZE), read_y_train(SIZE, SIZE)

    X_test, Y_test = read_x_test(SIZE, SIZE), read_y_test(SIZE, SIZE)

    if not os.path.isdir(DUMP_FOLDER_NAME):
        os.mkdir(DUMP_FOLDER_NAME)
    with open(f'{DUMP_FOLDER_NAME}/X_train.pkl', 'wb') as f:
        pickle.dump(X_train, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f'{DUMP_FOLDER_NAME}/Y_train.pkl', 'wb') as f:
        pickle.dump(Y_train, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f'{DUMP_FOLDER_NAME}/X_test.pkl', 'wb') as f:
        pickle.dump(X_test, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f'{DUMP_FOLDER_NAME}/Y_test.pkl', 'wb') as f:
        pickle.dump(Y_test, f, protocol=pickle.HIGHEST_PROTOCOL)
    print('saved')


X_train = X_train / 255.0
X_test = X_test / 255.0

if os.path.exists(f'{DUMP_FOLDER_NAME}/Y_train_one_hot.pkl'):
    with open(f'{DUMP_FOLDER_NAME}/Y_train_one_hot.pkl', 'rb') as f:
        Y_train_one_hot = pickle.load(f)
    with open(f'{DUMP_FOLDER_NAME}/Y_test_one_hot.pkl', 'rb') as f:
        Y_test_one_hot = pickle.load(f)
    print('loaded one hot')
else:
    Y_train_one_hot = masks_rgb_to_classes(Y_train)
    Y_test_one_hot = masks_rgb_to_classes(Y_test)
    print('saving one hot')
    with open(f'{DUMP_FOLDER_NAME}/Y_train_one_hot.pkl', 'wb') as f:
        pickle.dump(Y_train_one_hot, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f'{DUMP_FOLDER_NAME}/Y_test_one_hot.pkl', 'wb') as f:
        pickle.dump(Y_test_one_hot, f, protocol=pickle.HIGHEST_PROTOCOL)
    print('saved one hot')

# print(Y_train_one_hot.shape)
# print(masks_classes_to_rgb(Y_train_one_hot).shape)

plt.figure()
plt.subplot(1, 3, 1)
plt.imshow(X_train[0])
plt.subplot(1, 3, 2)
plt.imshow(Y_train[0] / 255)
plt.subplot(1, 3, 3)
plt.imshow(masks_classes_to_rgb(np.array([Y_train_one_hot[0]]))[0] / 255)
plt.show()

test_sample_x = X_test[0]
test_sample_y = Y_test[0]

with tf.device('/device:GPU:0'):
    model = create_model_and_compile(SIZE, SIZE, len(labels))

    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True),
        ModelCheckpoint("../cs-checkpoint.h5", save_best_only=True),
        ShowProgress(test_sample_x, test_sample_y)
    ]

    # Train The Model
    model.fit(
        X_train, Y_train_one_hot,
        validation_data=(X_test, Y_test_one_hot),
        epochs=50,
        callbacks=callbacks
    )