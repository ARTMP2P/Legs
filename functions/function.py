import os
import cv2
import numpy as np
from .input_data import *


def read_img(dir):
    """
    Считывает изображение из указанного каталога, изменяет его размер до 1024x1024,
    применяет к нему определенные операции обработки изображений: преобразование в
    бинарное изображение (значения 0 и 1) и приведение к формату int8. В конце функция
    возвращает обработанное изображение, увеличенное на одно измерение, а также ширину и высоту
    исходного изображения.

    :param dir: Путь к изображению.
    :return: Обработанное изображение с размером 1024x1024.
    """

    img = cv2.imread(dir, 0)  # Считываем изображение как черно-белое (grayscale)
    img = cv2.resize(img, (1024, 1024)).astype(np.bool_).astype(np.int8)  # Изменяем размер изображения на 1024x1024
    img[img == 0] = -1  # Преобразуем значения пикселей 0 в -1 (белый цвет)
    return np.expand_dims(img, axis=2)  # Увеличиваем изображение на одно измерение


def get_img_for_predict(dir_folder):
    list_img = []
    files = os.listdir(dir_folder)
    files.sort()
    for filename in files:
        if filename.endswith('.png'):
            list_img.append(read_img(os.path.join(dir_folder, filename)))

    X1 = np.concatenate(list_img, axis=-1)
    return np.expand_dims(X1, axis=0)  # shape np.array (1, 2048, 2048, 8)


def predict_img(img, side: str):
    if side == "mirr1":
        g_model = g_model1
    else:
        g_model = g_model2
    img = g_model.predict(img)[0] * 127.5 + 127.5
    return np.where(img < 128, 0, 255)  # shape np.array (2048, 2048, 8)


def save_gen_img(img, path, width, height):
    Z = np.zeros((8, 1024, 1024))
    for i in range(8):
        Z[i, :, :] = img[:, :, i]
        img = cv2.resize(Z[i], (width, height))
        # ==================================
        # Добавлена пробная функция, при необходимости можно отключить
        img = get_contour(img)
        # ==================================
        cv2.imwrite(os.path.join(path, f'{str(i)}.png'), img)


def get_image_size(image_path):
    """
    Считывает размеры изображения (ширину и высоту) из указанного пути.

    :param image_path: Путь к изображению.
    :return: Ширина и высота изображения.
    """
    img = cv2.imread(image_path)
    height, width = img.shape[:2]
    return width, height


def get_contour(img, arg_1=15, arg_2=15, thickness=25):  # arg_1 только нечетное значение
    color = 127

    thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, arg_1, arg_2)
    img[img > 127], img[img <= 127] = 255, 0
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    im = cv2.drawContours(image=thresh, contours=contours, contourIdx=-1, color=color, thickness=thickness)

    return im
