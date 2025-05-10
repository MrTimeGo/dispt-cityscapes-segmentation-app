import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

from keras.callbacks import EarlyStopping, ModelCheckpoint

from .model_instances.show_progress import ShowProgress
from .utils.image_reader import read_x_train, read_y_train, read_x_test, read_y_test
from .utils.mask_rgb_to_classes import masks_rgb_to_classes, labels, masks_classes_to_rgb
from .model_instances.u_net_a import create_model_and_compile

SIZE = 256
DUMP_FOLDER_NAME = '../image-loaded-checkpoints'

if os.path.exists(f'{DUMP_FOLDER_NAME}/X_train.npy'):
    X_train = np.load(f'{DUMP_FOLDER_NAME}/X_train.npy', allow_pickle=True)
    Y_train = np.load(f'{DUMP_FOLDER_NAME}/Y_train.npy', allow_pickle=True)
    X_test = np.load(f'{DUMP_FOLDER_NAME}/X_test.npy', allow_pickle=True)
    Y_test = np.load(f'{DUMP_FOLDER_NAME}/Y_test.npy', allow_pickle=True)
    print('loaded')
else:
    X_train, Y_train = read_x_train(SIZE, SIZE), read_y_train(SIZE, SIZE)

    X_test, Y_test = read_x_test(SIZE, SIZE), read_y_test(SIZE, SIZE)

    if not os.path.isdir(DUMP_FOLDER_NAME):
        os.mkdir(DUMP_FOLDER_NAME)

    X_train.dump(f'{DUMP_FOLDER_NAME}/X_train.npy')
    Y_train.dump(f'{DUMP_FOLDER_NAME}/Y_train.npy')
    X_test.dump(f'{DUMP_FOLDER_NAME}/X_test.npy')
    Y_test.dump(f'{DUMP_FOLDER_NAME}/Y_test.npy')
    print('saved')


X_train = X_train / 255.0
X_test = X_test / 255.0

if os.path.exists(f'{DUMP_FOLDER_NAME}/Y_train_one_hot.npy'):
    Y_train_one_hot = np.load(f'{DUMP_FOLDER_NAME}/Y_train_one_hot.npy')
    Y_test_one_hot = np.load(f'{DUMP_FOLDER_NAME}/Y_test_one_hot.npy')
else:
    Y_train_one_hot = masks_rgb_to_classes(Y_train)
    Y_test_one_hot = masks_rgb_to_classes(Y_test)

    Y_train_one_hot.dump(f'{DUMP_FOLDER_NAME}/Y_train_one_hot.npy')
    Y_test_one_hot.dump(f'{DUMP_FOLDER_NAME}/Y_test_one_hot.npy')

plt.figure()
plt.subplot(1, 3, 1)
print(X_train[0].shape)
plt.imshow(X_train[0])
plt.subplot(1, 3, 2)
plt.imshow(Y_train[0] / 255)
plt.subplot(1, 3, 3)
plt.imshow(masks_classes_to_rgb(np.array([Y_train_one_hot[0]]))[0] / 255)
plt.show()


with tf.device('/device:GPU:0'):
    model = create_model_and_compile(SIZE, SIZE, len(labels))

    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True),
        ModelCheckpoint("../cs-checkpoint.h5", save_best_only=True),
        ShowProgress(X_test, Y_test)
    ]

    # Train The Model
    model.fit(
        X_train, Y_train_one_hot,
        validation_data=(X_test, Y_test_one_hot),
        epochs=50,
        callbacks=callbacks
    )