import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot
import tensorflow as tf
from PIL import Image, ImageChops
from statistics import mean
import cv2
from builtins import range

# ======================================================================
from .vars import *
from .file_load import *
from .model_ import *

import os
import numpy as np
import random
import tensorflow as tf

os.environ['PYTHONHASHSEED'] = str(42)
random.seed(42)
tf.random.set_seed(42)
np.random.seed(42)


# ======================================================================


def summarize_performance(step, g_model, side_param, f=0):
    """
    Эта функция предназначена для сохранения обученных моделей и оценки их производительности на тестовых данных.
    Функция принимает в качестве аргументов номер шага обучения, обученную модель и флаг f, который управляет
    именем файла модели при сохранении. Если f равно 1, то имя файла будет содержать дополнительную метку 'good',
    а если f равно 0, то метка 'good' не добавляется.
    Функция использует сохраненную модель для генерации изображений на основе тестовых данных. Затем функция сравнивает
    сгенерированные изображения с оригинальными изображениями тестовых данных и вычисляет процентное отличие между ними.
    Результаты сохраняются в виде изображений в папке img_test, а также выводятся на экран в виде среднего значения
    процентного отличия для каждого тестового изображения.
    :param step: Номер шага обучения.
    :param g_model: Обученная модель.
    :param f: Флаг для имени файла модели (0 или 1).
    :param side_param: Значение стороны (left или right).
    """
    if f:
        filename_model_NN = f'{dir_model_NN}/M_good{step}.h5'
        g_model.save(filename_model_NN)
        print('> Saved:', filename_model_NN)
    else:
        filename_model_NN = f'{dir_model_NN}/M_{step}.h5'
        g_model.save(filename_model_NN)
        print('> Saved:', filename_model_NN)

    try:
        X, y_true = create_dataset(get_file_paths(root, side_param), 1, side_param)
        X = g_model.predict(np.array(X))
    except:
        print('Ошибка при создании датасета или генерации изображений')

    plist = []
    for j, im in enumerate(X):
        percentage_list = []

        for i in range(CHANEL):
            IMG = np.concatenate((np.expand_dims(im[:, :, i] * 255, 2),
                                  np.expand_dims(y_true[j][:, :, i] * 255, 2), np.zeros((SIZE, SIZE, 1))),
                                 axis=-1)

            differ = cv2.absdiff(y_true[j][:, :, i].astype(np.float64), im[:, :, i].astype(np.float64))
            differ = differ.astype(np.uint8)
            percentage = (np.count_nonzero(differ) * 100) / differ.size
            percentage_list.append(percentage)

            IMG_res = cv2.resize(IMG, (int(SIZE * 2), int(SIZE * 2)), interpolation=cv2.INTER_NEAREST)
            IMG_res = IMG_res[:, :, ::-1]
            cv2.imwrite(f'{img_test_group}/{rakurs[i]}_{j}.jpg', np.uint8(IMG_res))
            print(f"Percentage for {rakurs[i]} is: {round(percentage, 2)}")

        plist.append(mean(percentage_list))
        print(f"Mean percentage for model {j} is: {round(mean(percentage_list), 2)}")
    print(f"Mean percentage for all models is: {round(mean(plist), 2)}")
    with open(log_file, 'a+') as file:
        file.write(f'{filename_model_NN}\nMetricks: {mean(plist)}\n')


def train(d_model, g_model, gan_model, root_dir, n_epochs=200, n_batch=1, side="Right", bufer=0, save_plot=True):
    """
    Данная функция предназначена для обучения модели генеративно-состязательной сети (GAN) для задачи
    переноса стиля между изображениями. Она принимает на вход модели дискриминатора (d_model),
    генератора (g_model) и GAN (gan_model), количество эпох (n_epochs), размер пакета (n_batch),
    индекс начальной эпохи (i_s) и буфер (bufer).
    Внутри функции происходит генерация настоящих и фейковых изображений, обновление параметров
    дискриминатора и генератора, а также подсчет и вывод результатов обучения. Функция также вызывает
    вспомогательные функции для вывода статистики и результатов обучения.
    """
    shown_statics()
    i_s = 0
    # determine the output square shape of the discriminator
    n_patch = d_model.output_shape[1]
    print(f'n_patch  {n_patch}\nn_epochs = {n_epochs}')

    # Lists to store losses for plotting
    d1_losses = []
    d2_losses = []
    g_losses = []

    for i in range(n_steps):  # n_steps

        # list_A, list_B, list_y = [], [], []
        list_file_paths = get_file_paths(root_dir, side)
        print(n_batch, type(n_batch))
        batch_x, batch_y = create_dataset(list_file_paths, n_batch, side)
        print(len(batch_x), len(batch_y))

        X_realA = np.stack(batch_x, axis=0)
        X_realB = np.stack(batch_y, axis=0)
        y_real = np.ones((X_realA.shape[0], n_patch, n_patch, 1))

        print(f"X_realA {X_realA.shape}\n"
              f"X_realB {X_realB.shape}\n"
              f"y_real {y_real.shape}")

        X_fakeB, y_fake = generate_fake_samples(g_model, X_realA, n_patch)

        d_loss1 = d_model.train_on_batch([X_realA, X_realB], y_real)

        # update discriminator for generated samples
        d_loss2 = d_model.train_on_batch([X_realA, X_fakeB], y_fake)

        # update the generator
        g_loss, _, _ = gan_model.train_on_batch(X_realA, [y_real, X_realB])

        # Append losses to lists
        d1_losses.append(d_loss1)
        d2_losses.append(d_loss2)
        g_losses.append(g_loss)

        # summarize performance
        print('>%d, d1[%.3f] d2[%.3f] g[%.3f]' % (i + 1, d_loss1, d_loss2, g_loss))
        # summarize model performance
        if (i + 1) % (n_epochs) == 0:  # bat_per_epo // 200
            bufer = i
            i_s += 1
            summarize_performance(i_s, g_model, side)

        if (g_loss < 0.90) and (i - bufer > 25):
            bufer = i
            i_s += 1
            summarize_performance(i_s, g_model, side, f=1)

            # Save the plot of losses
        if save_plot:
            save_loss_plot(d1_losses, d2_losses, g_losses)


def save_loss_plot(d1_losses, d2_losses, g_losses):
    """
    Сохраняет график ошибок в файл.

    :param epoch: Номер эпохи.
    :param d1_losses: Список значений потерь для дискриминатора 1.
    :param d2_losses: Список значений потерь для дискриминатора 2.
    :param g_losses: Список значений потерь для генератора.
    """
    plt.plot(d1_losses, label='D1 Loss')
    plt.plot(d2_losses, label='D2 Loss')
    plt.plot(g_losses, label='G Loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('GAN Training Loss')
    plt.savefig(os.path.join(root.split("/")[0], 'loss_plot_epoch.png'))
    plt.close()


def shown_statics():
    print(f'Lerning rate: {lr}\nBatch: {batch}')


if __name__ == '__main__':
    pass
