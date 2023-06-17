import random

from .vars import *
import cv2
import os
import numpy as np
from numpy import ones
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# ======================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ImageBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = []

    def add_images(self, images):
        self.buffer.extend(images)
        if len(self.buffer) > self.buffer_size:
            self.buffer = self.buffer[-self.buffer_size:]

    def get_random_images(self, num_images):
        return random.sample(self.buffer, num_images)


def read_img(dir):
    """
    Эта функция считывает изображение из указанного каталога и изменяет его размер до заданного
    значения. Затем она применяет к нему определенные операции обработки изображений: обрезание,
    преобразование в бинарное изображение и приведение к формату int8. В конце функция возвращает
    изображение, увеличенное на одно измерение, чтобы сделать его совместимым с другими массивами,
    используемыми в коде.
    """

    img = cv2.imread(dir, 0).astype(np.bool_).astype(np.int8)
    img[img == 0] = -1
    return np.expand_dims(img, axis=2)


def get_name_model(s):
    """
    Функция get_name_model(s) принимает на вход строку s, которая содержит путь к файлу. Функция 
    ищет в строке s подстроку yaw_ и находит позицию первого символа этой подстроки. Затем функция 
    ищет первое вхождение символа / после позиции подстроки yaw_. Далее функция ищет позицию символа 
    .pos и использует его как индекс конца имени файла. Функция возвращает подстроку между начальной 
    и конечной позициями в строке s, содержащую имя файла с расширением .pos. Таким образом, функция 
    возвращает имя файла из пути к файлу.
    """
    # end = s.find('.pos')
    # start = s.find('/', s.find('yaw_'))

    m_name = s.split('/')[0]
    return m_name  # s[start + 1:end] + '.pos'


# Определение генератора
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # Реализуйте архитектуру вашего генератора
        # Пример:
        self.encoder = nn.Sequential(
            nn.Conv2d(8, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 8, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# Определение дискриминатора
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # Реализация архитектуры дискриминатора
        self.model = nn.Sequential(
            nn.Conv2d(8, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, kernel_size=4, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


# Define the combined generator and discriminator model for updating the generator
def define_gan(in_src: list, g_model, d_model, y):
    in_src = torch.randn(in_src[1:])
    print(f"{in_src.shape}\n{y.shape}")
    # Устанавливаем требуемые размерности
    in_channels, in_h, in_w = in_src.shape[0], in_src.shape[1], in_src.shape[2]
    out_channels = d_model(in_src.unsqueeze(0), y.unsqueeze(0)).shape[1]  # Получаем размерность выхода дискриминатора


    # Создаем модель GAN
    gan_model = nn.Sequential(
        g_model,
        d_model
    )

    # Определяем оптимизатор и функцию потерь
    optimizer = optim.Adam(gan_model.parameters(), lr=0.0002, betas=(0.5, 0.999))
    loss_fn = nn.BCELoss()

    return gan_model, optimizer, loss_fn


# def generate_real_samples(list_dir_name, list_dir_name_25, patch_shape, batch):
#     """
#     Функция generate_real_samples генерирует реальные образцы данных, которые используются для обучения
#     дискриминатора. Функция принимает список имен файлов и соответствующий им список файлов с масками,
#     загружает изображения, объединяет их в единую матрицу и добавляет дополнительную размерность. Функция возвращает
#     два изображения и метки классов (все классы помечены как реальные).
#     :param list_dir_name:
#     :param list_dir_name_25:
#     :param patch_shape:
#     :return:
#     """
#
#     list_img1 = list(map(read_img, list_dir_name))
#     X1 = np.concatenate(list_img1, axis=-1)
#     X1 = np.expand_dims(X1, axis=0)
#     list_img2 = list(map(read_img, list_dir_name_25))
#     X2 = np.concatenate(list_img2, axis=-1)
#     X2 = np.expand_dims(X2, axis=0)
#
#     # generate 'real' class labels (1)
#     y = ones((batch, 8, patch_shape, patch_shape))
#     return [X1, X2], y
#
#
# # Generate a batch of images, returns images and targets
# def generate_fake_samples(g_model, samples, batch_size, patch_shape):
#     with torch.no_grad():
#         X = g_model(samples)
#         X = X[:batch_size]  # Ограничиваем тензор X до нужного размера
#         y = torch.zeros((batch_size, 8, patch_shape, patch_shape))
#
#     return X, y

def generate_real_samples(list_dir_name, list_dir_name_25, patch_shape, batch):
    list_img1 = list(map(read_img, list_dir_name))
    X1 = np.concatenate(list_img1, axis=-1)
    X1 = np.transpose(X1, (2, 0, 1))  # Изменение порядка осей для PyTorch
    X1 = torch.from_numpy(X1).unsqueeze(0)

    list_img2 = list(map(read_img, list_dir_name_25))
    X2 = np.concatenate(list_img2, axis=-1)
    X2 = np.transpose(X2, (2, 0, 1))  # Изменение порядка осей для PyTorch
    X2 = torch.from_numpy(X2).unsqueeze(0)

    y = torch.ones((batch, 8, patch_shape, patch_shape))
    return [X1, X2], y


def generate_fake_samples(g_model, samples, batch_size, patch_shape):
    with torch.no_grad():
        X = g_model(samples)
        X = X[:batch_size]  # Ограничиваем тензор X до нужного размера
        y = torch.zeros((batch_size, 8, patch_shape, patch_shape))

    return X, y


# ======================================================================


# ======================================================================

if __name__ == '__main__':
    image_shape = [batch, CHANEL, SIZE, SIZE]
    Unet = UNet()
    print(f"Generator OUTPUT: {type(Unet)}")

