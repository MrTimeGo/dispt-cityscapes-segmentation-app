import numpy as np

class Label:
    def __init__(self, class_name: str, color: np.ndarray):
        self.class_name = class_name
        self.color = color


