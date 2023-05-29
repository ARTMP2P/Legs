from vars import *
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
        self.conv1 = torch.nn.Conv2d(in_image.shape[1], 64,
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
class EncoderBlock(nn.Module):
    def __init__(self, input_tensor, channels, batchnorm=True):
        super(EncoderBlock, self).__init__()

        self.input_tensor = input_tensor
        self.channels = channels
        self.batchnorm = batchnorm

        self.conv2d = nn.Conv2d(self.input_tensor.size()[1], self.channels,
                                kernel_size=3, stride=2, padding=1, bias=False)
        self.batchnorm2d = nn.BatchNorm2d(self.channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self):
        x = self.conv2d(self.input_tensor)
        nn.init.normal_(x, mean=0.0, std=0.02)
        # BatchNorm + ReLU
        if self.batchnorm:
            x = self.batchnorm2d(x)
            x = self.relu(x)

        return torch.tensor(x)


# Define a decoder block
class DecoderBlock(nn.Module):
    def __init__(self, input_tensor, concat_tensor, channels, dropout=True):
        super(DecoderBlock, self).__init__()

        self.input_tensor = input_tensor
        self.concat_tensor = concat_tensor
        self.channels = channels
        self.dropout = dropout
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv2d1 = nn.Conv2d(self.input_tensor.shape[1], self.channels,
                                kernel_size=1, stride=1, padding=0, bias=False)
        self.batchnorm2d = nn.BatchNorm2d(self.num_features)
        self.relu = nn.ReLU(inplace=True)
        self.dropout_layer = nn.Dropout(p=0.5)
        self.conv2d2 = nn.Conv2d(self.channels, self.channels,
                                kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self):
        x = self.upsample(self.input_tensor)
        decoded_tensor = nn.functional.interpolate(x, size=(self.concat_tensor.shape[2], self.concat_tensor.shape[3]))
        # Concatenate
        x = torch.cat([decoded_tensor, self.concat_tensor], dim=1)
        # 1x1 Convolutional Layer
        x = self.conv2d1(x)
        nn.init.normal_(x, mean=0.0, std=0.02)
        # BatchNorm + ReLU
        if self.dropout:
            x = self.batchnorm2d(x)
            x = self.relu(x)
            x = self.dropout_layer(x)
        # 3x3 Convolutional Layer
        x = self.conv2d2(x)
        nn.init.normal_(x, mean=0.0, std=0.02)
        # BatchNorm + ReLU
        x = self.batchnorm2d(x)
        x = self.relu(x)
        x = self.dropout_layer(x)

        return torch.tensor(x)


class UNetDownModule(nn.Module):
    def __init__(self, in_image, out_channels):
        super(UNetDownModule, self).__init__()

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_image.shape[0], out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.double_conv(x)
        return x


class Generator(nn.Module):
    def __init__(self, image_shape):
        super(Generator, self).__init__()

        # Image input
        self.in_image = torch.zeros(image_shape)

        # Encoder model
        self.e1 = EncoderBlock(self.in_image, 64, batchnorm=True)
        self.e2 = EncoderBlock(self.e1, 128)
        self.e3 = EncoderBlock(self.e2, 256)
        self.e4 = EncoderBlock(self.e3, 512)
        self.e5 = EncoderBlock(self.e4, 512)
        self.e6 = EncoderBlock(self.e5, 512)
        self.e7 = EncoderBlock(self.e6, 512)

        self.b = UNetDownModule(self.in_image, 512)

        # Decoder model
        self.d1 = DecoderBlock(self.b, self.e7, 512)
        self.d2 = DecoderBlock(self.d1, self.e6, 512)
        self.d3 = DecoderBlock(self.d2, self.e5, 512)
        self.d4 = DecoderBlock(self.d3, self.e4, 512, dropout=False)
        self.d5 = DecoderBlock(self.d4, self.e3, 256, dropout=False)
        self.d6 = DecoderBlock(self.d5, self.e2, 128, dropout=False)
        self.d7 = DecoderBlock(self.d6, self.e1, 64, dropout=False)

        # Output
        self.g = nn.ConvTranspose2d(self.d7.shape[1], CHANEL, kernel_size=4, stride=2, padding=1, bias=False)
        self.out_image = self.g(self.d7)
        nn.init.normal_(self.g.weight, mean=0.0, std=0.02)
        self.out_image = nn.Tanh()(self.out_image)

    def forward(self, x):
        # Encoder
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        e5 = self.e5(e4)
        e6 = self.e6(e5)
        e7 = self.e7(e6)

        # Reduce number of channels
        reduce_conv = self.conv_reduce(x)

        # Bottleneck
        bottleneck = self.b(reduce_conv)

        # Decoder
        d1 = self.d1(bottleneck, e7)
        d2 = self.d2(d1, e6)
        d3 = self.d3(d2, e5)
        d4 = self.d4(d3, e4)
        d5 = self.d5(d4, e3)
        d6 = self.d6(d5, e2)
        d7 = self.d7(d6, e1)

        # Output
        out = self.g(d7)
        out = nn.Tanh()(out)

        return out


# Define the combined generator and discriminator model for updating the generator
def define_gan(g_model, d_model, image_shape):
    """
    Function to define and return the GAN (Generative Adversarial Network) model, which consists of
    the generator (g_model) and discriminator (d_model). It combines the generator and discriminator into
    a single model and compiles it. Adam optimizer is used for training the model with loss weights [1, 100].
    The input layer of the model is the source image (in_src), and the output layers are the discrimination result
    and the generated image (dis_out and gen_out).
    Args:
        g_model: Generator model.
        d_model: Discriminator model.
        image_shape: Shape of the input image tensor.

    Returns:
        GAN model.
    """
    # Make weights in the discriminator not trainable
    for param in d_model.parameters():
        param.requires_grad = False

    # Define the source image
    in_src = torch.zeros(image_shape)

    # Connect the source image to the generator input
    gen_out = g_model(in_src)

    # Connect the source input and generator output to the discriminator input
    dis_out = d_model(torch.cat([in_src, gen_out], dim=1))

    # Source image as input, generated image and classification output
    model = nn.Model(in_src, [dis_out, gen_out])

    # Define optimizer and loss function
    opt = optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999))
    loss_fn = [nn.BCELoss(), nn.L1Loss()]

    return model


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
    in_image = torch.zeros(image_shape)
    downBlock = EncoderBlock(in_image, 512)
    print(f"Generator OUTPUT: {type(downBlock)}")
    convBlock = UNetDownModule(in_image, 512)
    print(f"Generator OUTPUT: {type(convBlock)}")
    upBlock = DecoderBlock(convBlock, downBlock, 512)
    print(f"Generator OUTPUT: {type(upBlock)}")
