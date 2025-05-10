import keras
import numpy as np
import matplotlib.pyplot as plt

from model.utils.mask_rgb_to_classes import masks_classes_to_rgb


class ShowProgress(keras.callbacks.Callback):

    def __init__(self, X_test, Y_test):
        self.X_test =  X_test
        self.Y_test = Y_test

        super(ShowProgress, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        id = np.random.randint(len(self.X_test))
        rand_img = self.X_test[id][np.newaxis, ...] # (1, w, h, 3)
        pred_mask = self.model.predict(rand_img)[0]  # (w, h, N-class)
        true_mask = self.Y_test[id] # (w, h, 3)

        pred_mask = masks_classes_to_rgb(np.array([pred_mask]))[0] # (w, h, 3)

        plt.subplot(1, 3, 1)
        plt.imshow(rand_img[0])
        plt.title("Original Image")
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(pred_mask)
        plt.title("Predicted Mask")
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(true_mask / 255)
        plt.title("True Mask")
        plt.axis('off')

        plt.tight_layout()
        plt.show()