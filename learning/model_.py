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


def read_img(dir):
    """
    Эта функция считывает изображение из указанного каталога и изменяет его размер до заданного
    значения. Затем она применяет к нему определенные операции обработки изображений: обрезание,
    преобразование в бинарное изображение и приведение к формату int8. В конце функция возвращает
    изображение, увеличенное на одно измерение, чтобы сделать его совместимым с другими массивами,
    используемыми в коде.
    """

    # try:
    #     img = cv2.imread(dir, 0)[y1:y2, x1:x2].astype(np.bool_).astype(np.int8)
    # except:
    #     print('read_img(dir)=', dir)
    # # img[:960] = -1
    # img[img == 0] = -1
    # img = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
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


class Discriminator(torch.nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()
        self.in_image = torch.zeros(input_shape)

        # Convolutional Layer
        self.conv1 = torch.nn.Conv2d(self.in_image.shape[1], 64,
                                     kernel_size=4, stride=2, padding=1, bias=False)
        self.relu1 = torch.nn.LeakyReLU(0.2, inplace=True)

        # Convolutional Layer + BatchNorm
        self.conv2 = torch.nn.Conv2d(64, 128,
                                     kernel_size=4, stride=2, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(num_features=128)
        self.relu2 = torch.nn.LeakyReLU(0.2, inplace=True)

        # Convolutional Layer + BatchNorm
        self.conv3 = torch.nn.Conv2d(128, 256,
                                     kernel_size=4, stride=2, padding=1, bias=False)
        self.bn3 = torch.nn.BatchNorm2d(num_features=256)
        self.relu3 = torch.nn.LeakyReLU(0.2, inplace=True)

        # Convolutional Layer + BatchNorm
        self.conv4 = torch.nn.Conv2d(256, 512,
                                     kernel_size=4, stride=1, padding=1, bias=False)
        self.bn4 = torch.nn.BatchNorm2d(num_features=512)
        self.relu4 = torch.nn.LeakyReLU(0.2, inplace=True)

    def forward(self):
        # Forward propagation
        out = self.conv1(self.in_image)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu3(out)
        out = self.conv4(out)
        out = self.bn4(out)
        out = self.relu4(out)

        # Output Layer
        out = torch.nn.Conv2d(512, 1,
                              kernel_size=4, stride=1,
                              padding=1, bias=False)(out)

        return torch.nn.Sigmoid()(out)


# Define an encoder block
class UNet(torch.nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        self.in_image = torch.zeros(input_shape)
        self.down1 = torch.nn.Sequential(
            # 1024x1024x8  => 512x512x64
            torch.nn.Conv2d(8, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )

        self.down2 = torch.nn.Sequential(
            # 512x512x64  => 256x256x128
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )

        self.down3 = torch.nn.Sequential(
            # 256x256x128  => 128x128x256
            torch.nn.Conv2d(128, 256, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )

        self.down4 = torch.nn.Sequential(
            # 128x128x256  => 64x64x512
            torch.nn.Conv2d(256, 512, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )

        self.down5 = torch.nn.Sequential(
            # 64x64x512  => 64x64x512
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )

        self.down6 = torch.nn.Sequential(
            # 64x64x512  => 64x64x512
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )

        self.down7 = torch.nn.Sequential(
            # 64x64x512  => 64x64x512
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )

        self.up1 = torch.nn.Sequential(
            # 64x64x512   => 64x64x512
            torch.nn.Upsample(scale_factor=2, mode='bilinear'),
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
            torch.nn.ReLU()
        )

        self.up2 = torch.nn.Sequential(
            # 64x64x512   => 64x64x512
            torch.nn.Upsample(scale_factor=2, mode='bilinear'),
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
            torch.nn.ReLU()
        )

        self.up3 = torch.nn.Sequential(
            # 64x64x512   => 64x64x512
            torch.nn.Upsample(scale_factor=2, mode='bilinear'),
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
            torch.nn.ReLU()
        )
        self.up4 = torch.nn.Sequential(
            # 64x64x512   => 128x128x256
            torch.nn.Upsample(scale_factor=2, mode='bilinear'),
            torch.nn.Conv2d(512, 256, kernel_size=3, padding=1),
            torch.nn.ReLU()
        )

        self.up5 = torch.nn.Sequential(
            # 128x128x256   => 256x256x128
            torch.nn.Upsample(scale_factor=2, mode='bilinear'),
            torch.nn.Conv2d(256, 128, kernel_size=3, padding=1),
            torch.nn.ReLU()
        )

        self.up6 = torch.nn.Sequential(
            # 256x256x128   => 512x512x64
            torch.nn.Upsample(scale_factor=2, mode='bilinear'),
            torch.nn.Conv2d(128, 64, kernel_size=3, padding=1),
            torch.nn.ReLU()
        )

        self.up7 = torch.nn.Sequential(
            # 512x512x64   => 1024x1024x8
            torch.nn.Upsample(scale_factor=2, mode='bilinear'),
            torch.nn.Conv2d(64, 8, kernel_size=3, padding=1),
            torch.nn.ReLU()
        )
    # Repeat the same pattern for the remaining 5 up layers

    def forward(self):
        out1 = self.down1(self.in_image)
        out2 = self.down2(out1)
        out3 = self.down3(out2)
        out4 = self.down4(out3)
        out5 = self.down5(out4)
        out6 = self.down6(out5)
        out7 = self.down7(out6)

        # Pass through the remaining 5 down layers

        out1 = self.up1(out7)
        out2 = self.up2(out1)
        out3 = self.up3(out2)
        out4 = self.up4(out3)
        out5 = self.up5(out4)
        out6 = self.up6(out5)
        out = self.up7(out6)

        # Pass through the remaining 5 up layers

        return out



# Define the combined generator and discriminator model for updating the generator
def get_model(input_array):
    # Create generator and discriminator models
    generator = UNet(input_array)
    discriminator = Discriminator(input_array)

    # Pass input through both models
    gen_output = generator()
    disc_output = discriminator()

    # Return output of discriminator with shape [8, 1024, 1024]
    return disc_output


def generate_real_samples(list_dir_name, list_dir_name_25, patch_shape):
    """
    Функция generate_real_samples генерирует реальные образцы данных, которые используются для обучения
    дискриминатора. Функция принимает список имен файлов и соответствующий им список файлов с масками,
    загружает изображения, объединяет их в единую матрицу и добавляет дополнительную размерность. Функция возвращает
    два изображения и метки классов (все классы помечены как реальные). :param list_dir_name: :param
    list_dir_name_25: :param n_samples: :param patch_shape: :return:
    """

    list_img1 = list(map(read_img, list_dir_name))
    X1 = np.concatenate(list_img1, axis=-1)
    X1 = np.expand_dims(X1, axis=0)
    list_img2 = list(map(read_img, list_dir_name_25))
    X2 = np.concatenate(list_img2, axis=-1)
    X2 = np.expand_dims(X2, axis=0)

    # generate 'real' class labels (1)
    y = ones((1, patch_shape, patch_shape, 1))
    return [X1, X2], y


# Generate a batch of images, returns images and targets
def generate_fake_samples(g_model, samples, patch_shape):
    with torch.no_grad():
        X = g_model(samples)
        y = torch.zeros((len(X), 1, patch_shape, patch_shape))

    return X, y


# ======================================================================


# ======================================================================

if __name__ == '__main__':
    image_shape = [batch, CHANEL, SIZE, SIZE]
    Unet = UNet()
    print(f"Generator OUTPUT: {type(Unet)}")

