import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot
import tensorflow as tf
from PIL import Image, ImageChops
from statistics import mean
import cv2

# ======================================================================
from .vars import *
from .file_load import *
from .model_ import *


# ======================================================================


def summarize_performance(step, g_model, f=0):
    """
    Эта функция предназначена для сохранения обученных моделей и оценки их производительности на тестовых данных. 
    Функция принимает в качестве аргументов номер шага обучения, обученную модель и флаг f, который управляет 
    именем файла модели при сохранении. Если f равно 1, то имя файла будет содержать дополнительную метку 'good', 
    а если f равно 0, то метка 'good' не добавляется.

    Функция использует сохраненную модель для генерации изображений на основе тестовых данных. Затем функция сравнивает 
    сгенерированные изображения с оригинальными изображениями тестовых данных и вычисляет процентное отличие между ними. 
    Результаты сохраняются в виде изображений в папке img_test, а также выводятся на экран в виде среднего значения 
    процентного отличия для каждого тестового изображения.    
    """
    if f:
        filename_model_NN = f'{dir_model_NN}/M_good{step}.h5'
        g_model.save(filename_model_NN)
        print('>Saved: %s' % filename_model_NN)
    else:
        filename_model_NN = f'{dir_model_NN}/M_{step}.h5'
        g_model.save(filename_model_NN)
        print('>Saved: %s' % filename_model_NN)

    """
    for n in range(len(model_test)): # len(model_test) - количество моделей ступней для теста
    val_1, val_2 = list_img(n, g_model, step)
    list_IMG.extend(val_1)
    list_metrics.extend(val_2)
    """

    print(f'list_img_test_array.shape= {list_img_test_array.shape}')
    try:
        X = g_model.predict(list_img_test_array)  # np.uint8()
    except:
        print('ошибка !!!')

    for j, im in enumerate(X):
        precentage_list = []
        # fig = plt.figure(figsize=(24, 11))
        print(dir_test[j][75:-20])
        for i in range(CHANEL):
            # try:
            # plt.subplot(2, CHANEL // 2, i + 1)
            # plt.title(list_metrics[i])
            # pyplot.axis('off')

            # print('333333', 'im[:,:,i]=', im[:,:,i].shape,  'list_img_test_25[j][:,:,i]=', list_img_test_25[j][:,:,i].shape)
            IMG = np.concatenate((np.expand_dims(im[:, :, i] * 255, 2),
                                  np.expand_dims(list_img_test_25[j][:, :, i] * 255, 2), np.zeros((SIZE, SIZE, 1))),
                                 axis=-1)
            # ============================================
            '''
                Вычисляет абсолютную разницу для каждого элемента между двумя массивами или между массивом и скаляром.
                '''
            image_eta = Image.fromarray(list_img_test_25[j][:, :, i], 'L')
            image_gan = Image.fromarray(im[:, :, i], 'L')

            differ = cv2.absdiff(list_img_test_25[j][:, :, i].astype(np.float64), im[:, :, i].astype(np.float64))
            differ = differ.astype(np.uint8)
            percentage = (np.count_nonzero(differ) * 100) / differ.size
            precentage_list.append(percentage)

            # ============================================
            IMG_res = cv2.resize(IMG, (int(SIZE * 2), int(SIZE * 2)), interpolation=cv2.INTER_NEAREST)
            IMG_res = IMG_res[:, :, ::-1]
            cv2.imwrite(f'{img_test_group}/{rakurs[i]}_{dir_test[j][75:-20]}.jpg', np.uint8(IMG_res))

        print(mean(precentage_list))
    with open(log_file, 'a+') as file:
        file.write(f'{filename_model_NN}\nMetricks: {mean(precentage_list)}\n')

    # train pix2pix models


def train(d_model, g_model, gan_model, dir, n_epochs=200, n_batch=1, i_s=0, bufer=0):
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

    for i in range(n_steps):  # n_steps

        list_A, list_B, list_y = [], [], []
        list_rand_dir, list_rand_dir_25 = get_list_dir_2(root, list_models, batch)
        # print(list_rand_dir, '\n', list_rand_dir_25)

        for i_d in range(batch):
            list_dir_name, list_dir_name_25 = list_rand_dir[i_d], list_rand_dir_25[i_d]
            [X_A, X_B], y = generate_real_samples(list_dir_name, list_dir_name_25, n_patch)
            list_A.append(X_A)
            list_B.append(X_B)
            list_y.append(y)

        X_realA = np.concatenate(list_A, axis=0)
        X_realB = np.concatenate(list_B, axis=0)
        y_real = np.concatenate(list_y, axis=0)

        X_fakeB, y_fake = generate_fake_samples(g_model, X_realA, n_patch)

        d_loss1 = d_model.train_on_batch([X_realA, X_realB], y_real)

        # update discriminator for generated samples
        d_loss2 = d_model.train_on_batch([X_realA, X_fakeB], y_fake)

        # update the generator
        g_loss, _, _ = gan_model.train_on_batch(X_realA, [y_real, X_realB])
        # summarize performance
        print('>%d, d1[%.3f] d2[%.3f] g[%.3f]' % (i + 1, d_loss1, d_loss2, g_loss))
        # summarize model performance
        if (i + 1) % (n_epochs) == 0:  # bat_per_epo // 200
            bufer = i
            i_s += 1
            summarize_performance(i_s, g_model)
            # break
        if (g_loss < 0.90) and (i - bufer > 25):
            bufer = i
            i_s += 1
            summarize_performance(i_s, g_model, f=1)
            # break


def shown_statics():
    print(f'Lerning rate: {lr}\nBatch: {batch}')


if __name__ == '__main__':
    pass
