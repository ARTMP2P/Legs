import os
import cv2
import numpy as np
from .input_data import *


def read_img(dir) -> np.array:
    img = cv2.imread(dir, cv2.IMREAD_GRAYSCALE)[800:2848, 136:2184].astype(np.bool_).astype(np.int8)
    img[img == 0] = -1
    return np.expand_dims(img, axis=2)  # shape np.array (2048, 2048, 1)


def get_img_for_predict(dir_folder):
    list_img = []
    files = os.listdir(dir_folder)
    files.sort()
    for filename in files:
        if filename.endswith('.png'):
            list_img.append(read_img(os.path.join(dir_folder, filename)))

    X1 = np.concatenate(list_img, axis=-1)
    return np.expand_dims(X1, axis=0)  # shape np.array (1, 2048, 2048, 8)


def predict_img(img):
    img = g_model.predict(img)[0] * 127.5 + 127.5
    return np.where(img < 128, 0, 255)  # shape np.array (2048, 2048, 8)


def save_gen_img(img, path):
    Z = np.zeros((8, 3088, 2320))
    for i in range(8):
        Z[i, 800:2848, 136:2184] = img[:, :, i]
        cv2.imwrite(os.path.join(path, f'{str(i)}.png'), Z[i])