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


def define_discriminator(input_shape):
    """
    Function to define a discriminator for the Pix-to-Pix torch model.

    Parameters:
        :param input_shape: (torch.Tensor): Input image
        list (N, C, H, W) where N is the number of images,
        C is the number of channels, H is the height of the image,
        and W is the width of the image by calling.
    Returns:
        :return discriminator (nn.Module): Discriminator network
        <class 'torch.Tensor'>
        Shape of OUTPUT: torch.Size([4, 1, 126, 126])
        Dtype of OUTPUT: torch.float32
    """
    in_image = torch.zeros(input_shape)
    # Convolutional Layer
    x = nn.Conv2d(in_image.shape[1], 64,
                  kernel_size=4, stride=2,
                  padding=1, bias=False)(in_image)
    nn.init.normal_(x, mean=0.0, std=0.02)
    x = nn.LeakyReLU(0.2, inplace=True)(x)

    # Convolutional Layer + BatchNorm
    x = nn.Conv2d(64, 128,
                  kernel_size=4, stride=2,
                  padding=1, bias=False)(x)
    nn.init.normal_(x, mean=0.0, std=0.02)
    x = nn.BatchNorm2d(num_features=128)(x)
    x = nn.LeakyReLU(0.2, inplace=True)(x)

    # Convolutional Layer + BatchNorm
    x = nn.Conv2d(128, 256,
                  kernel_size=4, stride=2,
                  padding=1, bias=False)(x)
    nn.init.normal_(x, mean=0.0, std=0.02)
    x = nn.BatchNorm2d(num_features=256)(x)
    x = nn.LeakyReLU(0.2, inplace=True)(x)

    # Convolutional Layer + BatchNorm
    x = nn.Conv2d(256, 512,
                  kernel_size=4, stride=1,
                  padding=1, bias=False)(x)
    nn.init.normal_(x, mean=0.0, std=0.02)
    x = nn.BatchNorm2d(num_features=512)(x)
    x = nn.LeakyReLU(0.2, inplace=True)(x)

    # Output
    x = nn.Conv2d(512, 1,
                  kernel_size=4, stride=1,
                  padding=1, bias=False)(x)
    return nn.Sigmoid()(x)


# Define an encoder block
def define_encoder_block(input_tensor, channels, batchnorm=True):
    """
    Function to define an encoder block for the Pix-to-Pix torch model.

    Parameters:
        :param input_tensor (torch.Tensor): Input tensor
        :param channels (int): Number of channels in output tensor
        :param batchnorm (bool): Whether to use batchnorm layers

    Returns:
        :return encoded_tensor (torch.Tensor): Encoded tensor
        <class 'torch.Tensor'>
        Shape of OUTPUT: torch.Size([4, 64, 512, 512])
        Dtype of OUTPUT: torch.float32
    """

    # 3x3 Convolutional Layer
    x = nn.Conv2d(input_tensor.shape[1], channels,
                  kernel_size=3, stride=2,
                  padding=1, bias=False)(input_tensor)
    nn.init.normal_(x, mean=0.0, std=0.02)

    # BatchNorm + ReLU
    if batchnorm:
        x = nn.BatchNorm2d(num_features=channels)(x)
        x = nn.ReLU(inplace=True)(x)

    return x


# Define a decoder block
def decoder_block(input_tensor, concat_tensor, channels, dropout=True):
    """
    Function to define a decoder block for the Pix-to-Pix torch model.

    Parameters:
        input_tensor (torch.Tensor): Input tensor
        concat_tensor (torch.Tensor): Tensor to concatenate with
        channels (int): Number of channels in output tensor
        dropout (bool): Whether to use dropout layers

    Returns:
        decoded_tensor (torch.Tensor): Decoded tensor
    """

    # Upsample
    x = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)(input_tensor)
    decoded_tensor = nn.functional.interpolate(x, size=(concat_tensor.shape[2], concat_tensor.shape[3]))

    # Concatenate
    x = torch.cat([decoded_tensor, concat_tensor], dim=1)

    # 1x1 Convolutional Layer
    x = nn.Conv2d(x.shape[1], channels,
                  kernel_size=1, stride=1,
                  padding=0, bias=False)(x)
    nn.init.normal_(x, mean=0.0, std=0.02)

    # BatchNorm + ReLU
    if dropout:
        x = nn.BatchNorm2d(num_features=channels)(x)
        x = nn.ReLU(inplace=True)(x)
        x = nn.Dropout(p=0.5)(x)

    # 3x3 Convolutional Layer
    x = nn.Conv2d(channels, channels,
                  kernel_size=3, stride=1,
                  padding=1, bias=False)(x)
    nn.init.normal_(x, mean=0.0, std=0.02)

    # BatchNorm + ReLU
    x = nn.BatchNorm2d(num_features=channels)(x)
    x = nn.ReLU(inplace=True)(x)
    x = nn.Dropout(p=0.5)(x)

    return x


# Define the standalone generator model
def define_generator(image_shape):
    """
    Function to define the architecture and return the generator model for image transformation.
    The generator uses convolutional layers to transform the input image into a hidden representation,
    and then uses decoder layers to transform the hidden representation into an output image of the same
    shape and size as the input image.
    Args:
        image_shape: Shape of the input image tensor.

    Returns:
        Generator model.
    """
    # Weight initialization
    init = nn.init.normal_
    # Image input
    in_image = torch.zeros(image_shape)

    # Encoder model
    e1 = define_encoder_block(in_image, 64, batchnorm=True)
    e2 = define_encoder_block(e1, 128)
    e3 = define_encoder_block(e2, 256)
    e4 = define_encoder_block(e3, 512)
    e5 = define_encoder_block(e4, 512)
    e6 = define_encoder_block(e5, 512)
    e7 = define_encoder_block(e6, 512)

    # 1x1 Convolutional Layer to reduce number of channels in input
    conv_reduce = nn.Conv2d(in_channels=in_image.shape[1],
                            out_channels=512,
                            kernel_size=3,
                            stride=2,
                            padding=1,
                            bias=False)
    nn.init.normal_(conv_reduce.weight, mean=0.0, std=0.02)

    # Set running_mean to have 512 elements
    conv_reduce.running_mean = [i * 0.02 for i in range(512)]

    # Apply 1x1 Convolutional Layer
    x = conv_reduce(in_image)

    # Bottleneck, no batch norm and ReLU
    b = nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1, bias=False)
    nn.init.normal_(b.weight, mean=0.0, std=0.02)

    # Add dimension
    b = b(x)

    # Apply ReLU
    b = nn.ReLU(inplace=True)(b)

    # Decoder model
    d1 = decoder_block(b, e7, 512)
    d2 = decoder_block(d1, e6, 512)
    d3 = decoder_block(d2, e5, 512)
    d4 = decoder_block(d3, e4, 512, dropout=False)
    d5 = decoder_block(d4, e3, 256, dropout=False)
    d6 = decoder_block(d5, e2, 128, dropout=False)
    d7 = decoder_block(d6, e1, 64, dropout=False)

    # Output
    g = nn.ConvTranspose2d(d7.shape[1], CHANEL, kernel_size=4, stride=2, padding=1, bias=False)
    out_image = g(d7)
    nn.init.normal_(g.weight, mean=0.0, std=0.02)
    out_image = nn.Tanh()(out_image)

    # Initialize and optimize model
    # model = init_and_optimize_model(model, image_shape)

    # Return the model
    # return model
    # Define model
    # model = nn.Sequential(
    #     in_image,
    #     e1, e2, e3, e4, e5, e6, e7,
    #     b,
    #     d1, d2, d3, d4, d5, d6, d7,
    #     out_image
    # )

    # return model


class Generator(nn.Module):
    def __init__(self, image_shape):
        super(Generator, self).__init__()

        # Image input
        self.in_image = torch.zeros(image_shape)

        # Encoder model
        self.e1 = define_encoder_block(self.in_image, 64, batchnorm=True)
        self.e2 = define_encoder_block(self.e1, 128)
        self.e3 = define_encoder_block(self.e2, 256)
        self.e4 = define_encoder_block(self.e3, 512)
        self.e5 = define_encoder_block(self.e4, 512)
        self.e6 = define_encoder_block(self.e5, 512)
        self.e7 = define_encoder_block(self.e6, 512)

        # 1x1 Convolutional Layer to reduce number of channels in input
        self.conv_reduce = nn.Conv2d(in_channels=self.in_image.shape[1],
                                     out_channels=512,
                                     kernel_size=3,
                                     stride=2,
                                     padding=1,
                                     bias=False)
        nn.init.normal_(self.conv_reduce.weight, mean=0.0, std=0.02)
        # Set running_mean to have 512 elements
        self.conv_reduce.running_mean = [i * 0.02 for i in range(512)]
        # x = self.conv_reduce(self.in_image)
        # Bottleneck, no batch norm and ReLU
        self.b = nn.Conv2d(512, 512,
                           kernel_size=4,
                           stride=2,
                           padding=1,
                           bias=False)
        nn.init.normal_(self.b.weight, mean=0.0, std=0.02)

        self.b = self.b(self.e7)
        # Add dimension
        # self.b = self.b(x)

        # Apply ReLU
        # self.b = nn.ReLU(inplace=True)(self.b)
        # print(f"Shape of OUTPUT: {self.b.shape}\nDtype of OUTPUT: {self.b.dtype}")
        # Decoder model
        self.d1 = decoder_block(self.b, self.e7, 512)
        self.d2 = decoder_block(self.d1, self.e6, 512)
        self.d3 = decoder_block(self.d2, self.e5, 512)
        self.d4 = decoder_block(self.d3, self.e4, 512, dropout=False)
        self.d5 = decoder_block(self.d4, self.e3, 256, dropout=False)
        self.d6 = decoder_block(self.d5, self.e2, 128, dropout=False)
        self.d7 = decoder_block(self.d6, self.e1, 64, dropout=False)

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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    image_shape = [batch, CHANEL, SIZE, SIZE]
    generator = Generator(image_shape)
    print(type(generator))
    print(f"Shape of OUTPUT: {generator.shape}\nDtype of OUTPUT: {generator.dtype}")
