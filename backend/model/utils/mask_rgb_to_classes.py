import warnings

import numpy as np

from .label import Label

labels = np.array([
    Label('void', np.array([0, 0, 0])),
    Label('road', np.array([128, 64, 128])),
    Label('sidewalk', np.array([244, 35, 232])),
    Label('person', np.array([220, 20, 60])),
    Label('rider', np.array([255, 0, 0])),
    Label('car', np.array([0, 0, 142])),
    Label('truck', np.array([0, 0, 70])),
    Label('bus', np.array([0, 60, 100])),
    Label('on rails', np.array([0, 80, 100])),
    Label('motorcycle', np.array([0, 0, 230])),
    Label('bicycle', np.array([119, 11, 32])),
    Label('building', np.array([70, 70, 70])),
    Label('wall', np.array([102, 102, 156])),
    Label('fence', np.array([190, 153, 153])),
    Label('pole', np.array([153, 153, 153])),
    Label('traffic sign', np.array([220, 220, 0])),
    Label('traffic light', np.array([250, 170, 30])),
    Label('vegetation', np.array([107, 142, 35])),
    Label('terrain', np.array([152, 251, 152])),
    Label('sky', np.array([70, 130, 180])),
    Label('rail track', np.array([81, 0, 81])),
    Label('parking', np.array([250, 170, 160])),
    Label('other', np.array([111, 74, 0])),
    Label('unknown1', np.array([0, 0, 110])),
    Label('unknown2', np.array([180, 165, 180])),
    Label('unknown3', np.array([150, 100, 100])),
    Label('unknown4', np.array([0, 0, 90])),
    Label('unknown5', np.array([150, 120, 90])),
    Label('unknown6', np.array([230, 150, 140])),
])

rgb_to_class = {
    tuple(label.color.tolist()): np.eye(len(labels), dtype=np.uint8)[i]
    for i, label in enumerate(labels)
}

# convert rbg masks (len, w, h, 3) to
# classes masks (len, w, h, n_classes)
def masks_rgb_to_classes(masks: np.ndarray) -> np.ndarray:
    n, w, h, _ = masks.shape
    n_classes = len(labels)
    output = np.zeros((n, w, h, n_classes), dtype=np.uint8)

    flat_images = masks.reshape(-1, 3)
    flat_output = output.reshape(-1, n_classes)

    rgb_keys = [tuple(pixel) for pixel in flat_images]

    for idx, rgb in enumerate(rgb_keys):
        if rgb in rgb_to_class:
            flat_output[idx] = rgb_to_class[rgb]
        else:
            #raise ValueError(f'unknown color {rgb}')
            print(f'Unknown color ({rgb[0]}, {rgb[1]}, {rgb[2]})')
            flat_output[idx] = rgb_to_class[(0, 0, 0)]

    return output


def masks_classes_to_rgb(class_array):
    n, w, h, n_classes = class_array.shape
    # Step 1: Convert one-hot vectors to class indices
    class_indices = np.argmax(class_array, axis=-1)  # shape (n, w, h)

    colormap = np.array([label.color for label in labels])

    output = colormap[class_indices]

    return output