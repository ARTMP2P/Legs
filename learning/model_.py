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


class Define_discriminator(nn.Module):
    """
    Определяет модель дискриминатора в архитектуре CycleGAN. На вход функции подаются
    два тензора, представляющие исходное изображение размерностью 8x1024x1024 и изображение целевого домена
    размерностью 8x1024x1024. Далее, изображения объединяются по каналам, и проходят через 5 слоев свертки, с каждым
    слоем количество фильтров увеличивается,
    а размер изображения уменьшается в два раза. Последний слой выдает карту признаков размером 1x1, которая проходит
    через сигмоиду, чтобы получить вероятность того, что пара изображений является настоящей. В результате
    возвращает скомпилированную модель дискриминатора.
    """
    def __init__(self):
        super(Define_discriminator, self).__init__()

        self.conv1 = nn.Conv2d(16, 32, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2)
        self.conv5 = nn.Conv2d(256, 1, kernel_size=3, stride=2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.tensor, y: torch.tensor):
        # concat input images along channels
        xy = torch.cat([x, y], dim=1)

        xy = self.conv1(xy)
        xy = self.conv2(xy)
        xy = self.conv3(xy)
        xy = self.conv4(xy)
        xy = self.conv5(xy)
        xy = self.sigmoid(xy)

        return xy


class DefineGenerator(torch.nn.Module):
    """
    определяет архитектуру и возвращает модель генератора для преобразования изображений.
    Генератор использует 7 сверточных слоев, заданных функцией encoder_block,
    (которая определяет блок кодировщика для модели генератора в архитектуре Pix2Pix.
    Она принимает на вход слой layer_in и число фильтров n_filters, которое определяет
    количество фильтров в сверточном слое. Параметр batchnorm определяет,
    следует ли применять слой нормализации по батчу.
    Функция добавляет сверточный слой с заданным числом фильтров и ядром
    размером (4, 4), используя метод инициализации весов RandomNormal)
    для преобразования входного изображения в скрытое представление,
    а затем использует 7 слоев декодирования, которые задаются функцией
    decoder_block (представляет собой один блок декодера модели
    генератора в архитектуре U-Net. Она принимает на вход тензор layer_in -
    входной слой блока декодера и тензор skip_in - соответствующий слой кодировщика.
    Затем функция добавляет слой Transposed Convolution для увеличения размерности
    входного тензора, применяет слой Batch Normalization, Dropout (если dropout=True)
    и выполняет операцию конкатенации со слоем кодировщика), чтобы преобразовать
    скрытое представление в выходное изображение той же формы и размера, что и входное изображение.
    Функция возвращает torch модель генератора.
    """

    def __init__(self, batch_size):
        super(DefineGenerator, self).__init__()
        self.batch_size = batch_size

        # Encoder layers
        self.encoder_blocks = nn.ModuleList()
        in_channels = 8
        for n_filters in [32, 64, 128, 256, 512, 512, 512]:
            self.encoder_blocks.append(self.encoder_block(in_channels, n_filters))
            in_channels = n_filters

        # Bottleneck layer
        self.bottleneck = self.encoder_block(in_channels, 512)

        # Decoder layers
        self.decoder_blocks = nn.ModuleList()
        out_channels = 512
        for n_filters in [512, 512, 512, 256, 128, 64, 32]:
            self.decoder_blocks.append(self.decoder_block(out_channels, out_channels // 2, n_filters))
            out_channels = n_filters

        # Output layer
        self.output_layer = nn.ConvTranspose2d(out_channels, 8, kernel_size=4, stride=2, padding=1, bias=False)
        self.output_layer.weight.data.normal_(0.0, 0.02)

    def encoder_block(self, in_channels, out_channels, batchnorm=True):
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False))
        if batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(negative_slope=0.2))
        return nn.Sequential(*layers)

    def decoder_block(self, in_channels, skip_channels, out_channels, dropout=True, batchnorm=True):
        layers = []

        # Upsampling layer
        layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1))
        if batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        if dropout:
            layers.append(nn.Dropout(p=0.5))
        layers.append(nn.ReLU())

        return nn.Sequential(*layers)

    def forward(self, input_img):
        # Encoder layers
        skip_connections = []
        layer_in = input_img
        for encoder_block in self.encoder_blocks:
            layer_out = encoder_block(layer_in)
            skip_connections.append(layer_out)
            layer_in = layer_out
            # print("Encoder layer output shape:", layer_out.shape)

        # Bottleneck layer
        bottleneck_out = self.bottleneck(layer_out)
        # print("Bottleneck layer output shape:", bottleneck_out.shape)

        # Decoder layers
        out_channels = 1024
        for decoder_block, skip_connection in zip(self.decoder_blocks, reversed(skip_connections)):
            layer_in = decoder_block(bottleneck_out)  # Развернуть тензор bottleneck_out

            layer_in = torch.cat((layer_in, skip_connection))  # Конкатенировать развернутый тензор и skip_connection
            # print("Decoder layer output shape:", layer_in.shape)
            bottleneck_out = layer_in

        output_img = self.output_layer(layer_in)
        output_img = torch.tanh(output_img)
        # print("Output image shape:", output_img.shape)

        return output_img


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(8, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 4, stride=2, padding=1),
            nn.ReLU(),
        )

        self.adapt = nn.Sequential(
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 512, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 512, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 512, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 8, 4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)  # Изменение порядка размерностей
        x = self.encoder(x)
        x = self.adapt(x)
        x = self.decoder(x)
        x = x[:, :, :8]  # Удаление последней размерности 4
        x = x.permute(0, 2, 3, 1)  # Удаление лишних размерностей
        return x


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

