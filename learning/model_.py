from keras.initializers import RandomNormal
from tensorflow.keras.optimizers import Adam

from .vars import *

from keras.initializers import RandomNormal
from keras.models import Model

from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Activation
from keras.layers import Concatenate
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import LeakyReLU
from keras.models import load_model
import cv2
import os
import numpy as np
from numpy import ones
from numpy import zeros


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
    print(f"Image {dir} loaded!!!")
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


# define the discriminator model
def define_discriminator(image_shape):
    """
    Функция define_discriminator определяет модель дискриминатора в архитектуре CycleGAN. На вход функции подаются
    два тензора, представляющие исходное изображение и изображение целевого домена. Далее, изображения объединяются
    по каналам, и проходят через несколько слоев свертки, с каждым слоем количество фильтров увеличивается,
    а размер изображения уменьшается в два раза. Последний слой выдает карту признаков размером 1x1, которая проходит
    через сигмоиду, чтобы получить вероятность того, что пара изображений является настоящей. В результате,
    функция возвращает скомпилированную модель дискриминатора. :param image_shape: :return:
    """
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # source image input
    in_src_image = Input(shape=image_shape)
    # target image input
    in_target_image = Input(shape=image_shape)
    # concatenate images channel-wise
    merged = Concatenate()([in_src_image, in_target_image])
    # C64
    d = Conv2D(64, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(merged)
    d = LeakyReLU(alpha=0.2)(d)
    # C128
    d = Conv2D(128, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # C256
    d = Conv2D(256, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # C512
    d = Conv2D(512, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # second last output layer
    d = Conv2D(512, (4, 4), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # patch output
    d = Conv2D(1, (4, 4), padding='same', kernel_initializer=init)(d)
    patch_out = Activation('sigmoid')(d)
    # define model
    model = Model([in_src_image, in_target_image], patch_out)
    # compile model
    opt = Adam(learning_rate=lr, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=[0.5])
    return model


# define an encoder block
def define_encoder_block(layer_in, n_filters, batchnorm=True):
    """
    Функция define_encoder_block определяет блок кодировщика для модели генератора в архитектуре Pix2Pix. Она
    принимает на вход слой layer_in и число фильтров n_filters, которое определяет количество фильтров в сверточном
    слое. Параметр batchnorm определяет, следует ли применять слой нормализации по батчу. Функция добавляет
    сверточный слой с заданным числом фильтров и ядром размером (4, 4), используя метод инициализации весов
    RandomNormal. Если параметр batchnorm установлен в True, функция также добавляет слой нормализации по батчу.
    Затем функция применяет активацию LeakyReLU с коэффициентом отрицательной области равным 0.2. Функция возвращает
    выходной слой. :param layer_in: :param n_filters: :param batchnorm: :return:
    """
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # add downsampling layer
    g = Conv2D(n_filters, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(layer_in)
    # conditionally add batch normalization
    if batchnorm:
        g = BatchNormalization()(g, training=True)
    # leaky relu activation
    g = LeakyReLU(alpha=0.2)(g)
    return g


# define a decoder block
def decoder_block(layer_in, skip_in, n_filters, dropout=True):
    """
    Функция decoder_block() представляет собой один блок декодера модели генератора в архитектуре U-Net. Она
    принимает на вход тензор layer_in - входной слой блока декодера и тензор skip_in - соответствующий слой
    кодировщика. Затем функция добавляет слой Transposed Convolution для увеличения размерности входного тензора,
    применяет слой Batch Normalization, Dropout (если dropout=True) и выполняет операцию конкатенации со слоем
    кодировщика. Затем функция применяет функцию активации ReLU и возвращает выходной тензор блока декодера. :param
    layer_in: :param skip_in: :param n_filters: :param dropout: :return:
    """
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # add upsampling layer
    g = Conv2DTranspose(n_filters, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(layer_in)
    # add batch normalization
    g = BatchNormalization()(g, training=True)
    # conditionally add dropout
    if dropout:
        g = Dropout(0.5)(g, training=True)
    # merge with skip connection
    g = Concatenate()([g, skip_in])
    # relu activation
    g = Activation('relu')(g)
    return g


# define the standalone generator model
def define_generator(image_shape):
    """
    Функция define_generator определяет архитектуру и возвращает модель генератора для преобразования изображений.
    Генератор использует сверточные слои для преобразования входного изображения в скрытое представление,
    а затем использует слои декодирования, чтобы преобразовать скрытое представление в выходное изображение той же
    формы и размера, что и входное изображение. Функция возвращает Keras модель генератора. :param image_shape:
    :return:
    """
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # image input
    in_image = Input(shape=image_shape)
    # encoder model
    e1 = define_encoder_block(in_image, 64, batchnorm=False)
    e2 = define_encoder_block(e1, 128)
    e3 = define_encoder_block(e2, 256)
    e4 = define_encoder_block(e3, 512)
    e5 = define_encoder_block(e4, 512)
    e6 = define_encoder_block(e5, 512)
    e7 = define_encoder_block(e6, 512)
    # bottleneck, no batch norm and relu
    b = Conv2D(512, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(e7)
    b = Activation('relu')(b)
    # decoder model
    d1 = decoder_block(b, e7, 512)
    d2 = decoder_block(d1, e6, 512)
    d3 = decoder_block(d2, e5, 512)
    d4 = decoder_block(d3, e4, 512, dropout=False)
    d5 = decoder_block(d4, e3, 256, dropout=False)
    d6 = decoder_block(d5, e2, 128, dropout=False)
    d7 = decoder_block(d6, e1, 64, dropout=False)
    # output
    g = Conv2DTranspose(CHANEL, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d7)
    out_image = Activation('tanh')(g)
    # define model
    model = Model(in_image, out_image)
    return model


# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model, image_shape):
    """
    Функция define_gan определяет и возвращает модель GAN (Generative Adversarial Network), которая состоит из
    генератора (g_model) и дискриминатора (d_model). Она объединяет генератор и дискриминатор в единую модель и
    компилирует ее. Оптимизатор Adam используется для обучения модели, веса потерь указаны как [1, 100]. Входным
    слоем модели является исходное изображение (in_src), а выходными слоями являются результат дискриминации и
    сгенерированное изображение (dis_out и gen_out). :param g_model: :param d_model: :param image_shape: :return:
    """
    # make weights in the discriminator not trainable
    for layer in d_model.layers:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = False
    # define the source image
    in_src = Input(shape=image_shape)
    # connect the source image to the generator input
    gen_out = g_model(in_src)
    # connect the source input and generator output to the discriminator input
    dis_out = d_model([in_src, gen_out])
    # src image as input, generated image and classification output
    model = Model(in_src, [dis_out, gen_out])
    # compile model
    opt = Adam(learning_rate=lr, beta_1=0.5)
    model.compile(loss=['binary_crossentropy', 'mae'], optimizer=opt, loss_weights=[1, 100])
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


# generate a batch of images, returns images and targets
def generate_fake_samples(g_model, samples, patch_shape):
    X = g_model.predict(samples)
    y = zeros((len(X), patch_shape, patch_shape, 1))

    return X, y


# ======================================================================


# ======================================================================

if __name__ == '__main__':
    pass
