# ======================================================================
from .vars import *
from .model_ import *
import os
import cv2
import random
import glob
import numpy as np
from tqdm import tqdm


# ======================================================================


# ======================================================================

def get_dirs(dir_file_txt):
    """
    Получает список директорий из текстового файла, удаляет символы "[" и "]"
    из каждой строки и возвращает список директорий.
    """
    dirs = []
    file_d = open(dir_file_txt, "r")
    for d in file_d:
        dirs.append(d[:-1])
    file_d.close()
    return dirs


def get_rand_index(dir):
    """
    Генерирует случайный индекс для выбора случайного изображения из списка директорий.
    """
    rand_index = list(range(len(dir)))
    np.random.shuffle(rand_index)
    return rand_index


def read_img25(dir):
    """
    читает изображение из директории, обрезает его, преобразует в черно-белое, 
    изменяет размер и возвращает его в виде массива numpy.

    Функция полностью дублирует функцию read_img, в виду отдельной предобработки датасета имеет смысл отказаться от нее
    """

    if os.path.exists(dir):
        img = cv2.imread(dir, 0).astype(np.bool_).astype(np.int8)
    else:
        print(f"Path {dir} exist: {os.path.exists(dir)}")
    return np.expand_dims(img, axis=2)


def get_list_dir(dir):
    """
    Возвращает список директорий для заданного файла и ракурсов.
    """
    # d = os.path.join(root, 'output_part' + dir + '_segmap.png')
    d = os.path.join(dir[:-len('_segmap.png')] + '_segmap.png')

    # d_25 = os.path.join(root, 'output_part' + dir[:i+1], '25_segmap.png')
    d_25 = os.path.join(dir[:-(len('_segmap.png') + 1)], '49_segmap.png')

    list_dir, list_dir_25 = [], []
    for r in rakurs:
        d_r = d.replace("yaw_0", r)
        list_dir.append(d_r)
        d_r_25 = d_25.replace("yaw_0", r)
        list_dir_25.append(d_r_25)
    return list_dir, list_dir_25


def get_list_dir_2(root, list_models, batch):
    '''
    получаем список рандомных дирректорий для выбранных модели, для каждого ракурса
    list(batch, list(dir-rakurs))
    '''
    list_rand_dir, list_rand_dir_25 = [], []
    for m in random.sample(list_models, batch):
        list_model_dir, list_model_dir_25 = [], []
        for r in rakurs:
            dir_rand = (random.choices(list(filter(lambda x: (m in x) & (r in x), dir_260_clear))))[0]
            index = dir_rand[-3:].find('/')
            dir_rand_25 = os.path.join(root, dir_rand[:index - 3], '49_segmap.png')
            dir_rand = os.path.join(root, dir_rand + '_segmap.png')
            list_model_dir.append(dir_rand)
            list_model_dir_25.append(dir_rand_25)
        list_rand_dir.append(list_model_dir)
        list_rand_dir_25.append(list_model_dir_25)

    return list_rand_dir, list_rand_dir_25


def get_dir_bug(s):
    """
    Возвращает подстроку из строки, содержащую информацию о модели и ракурсе.
    """
    end = s.find('24.hdf5')
    start = s.find('yaw_')
    return s[start:end]


def find_files_by_name(root_dir, file_name):
    """
    Описание изменений:

    Вместо итерации по каталогам с помощью os.walk, мы используем предопределенный список каталогов, который мы хотим
    обойти (список ['yaw_0', 'yaw_55', 'yaw_90', 'yaw_125', 'yaw_180', 'yaw_235', 'yaw_270', 'yaw_305']).
    Для каждого из этих каталогов мы снова используем os.walk, чтобы найти файлы с заданным именем.
    Вместо проверки подстроки в пути каталога, мы конкатенируем путь root_dir с каждым из предопределенных каталогов.
    Найденные пути к файлам добавляются в список file_paths.
    В конце функция возвращает отсортированный список найденных путей к файлам.
    :param root_dir:
    :param file_name:
    :return:
    """
    # список для хранения найденных путей к файлам
    file_paths = []

    # рекурсивно обойти все подкаталоги и найти файлы с заданным именем
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename == file_name and 'yaw_0' in dirpath:
                # если имя файла совпадает, добавить путь к файлу в список
                file_paths.append(os.path.join(dirpath, filename))

    # вернуть отсортированный список найденных путей к файлам
    return file_paths


# ======================================================================

list_img_test, list_img_test_25, list_img_train = [], [], []
dir = get_dirs(dir_file_txt)
dir_260_clear = get_dirs(dir_260_clear_file_txt)
dir_test = np.array(random.choices (find_files_by_name(img_test, '9_segmap.png'), k=8))
print(dir_test)
for d in dir_260_clear:
    name_model = get_name_model(d)
    if name_model not in list_models:
        list_models.append(name_model)

for d in dir_test:
    list_img_test.append(np.concatenate(list(map(read_img, ((get_list_dir(d))[0]))), axis=-1))
    list_img_test_25.append(np.concatenate(list(map(read_img, ((get_list_dir(d))[1]))), axis=-1))
# list_img_test = np.transpose(list_img_test, (3, 1, 2, 0))
print('list_img_test shape=', list_img_test[0].shape, 'list_img_test 49 shape=', list_img_test_25[0].shape)

# ======================================================================


def get_file_paths(root_dir: str) -> list:
    """

    :param root_dir:
    :return:
    """
    subfolder_list = sorted(os.listdir(root_dir))  # Список подпапок с номерами моделей
    file_paths = []

    for subfolder_model in subfolder_list:
        model_dir = os.path.join(root_dir, subfolder_model)

        for dirpath, _, filenames in os.walk(model_dir):
            for filename in filenames:
                if filename == "0_segmap.png":
                    if "yaw_0" in dirpath and os.path.exists(os.path.join(dirpath, '0_segmap.png')):
                        file_paths.append(dirpath)

    return file_paths


def create_dataset(file_paths: list, batch_size: int) -> list:
    """

    :param file_paths:
    :param batch_size:
    :return:
    """
    dataset = []
    for _ in tqdm(range(batch_size), desc="Processing files"):
        file_path = random.choice(file_paths)
        temp_x = []
        temp_y = []
        num_file = random.randint(0, 48)
        for m in ['0', '55', '90', '125', '180', '235', '270', '305']:
            d_file_path = file_path.replace('yaw_0', f"yaw_{m}")
            displacement_file_path = os.path.join(d_file_path, f"{num_file}_segmap.png")

            try:
                tensor = read_img(displacement_file_path)
                temp_x.append(tensor)
            except:
                print(displacement_file_path)

        for m in enumerate['0', '55', '90', '125', '180', '235', '270', '305']:
            t_file_path = file_path.replace('yaw_0', f"yaw_{m}")
            true_file_path = os.path.join(t_file_path, f"49_segmap.png")

            try:
                true_tensor = read_img(true_file_path)
                temp_y.append(true_tensor)
            except:
                print(true_file_path)

        dataset.append([np.concatenate(temp_x, axis=-1), np.concatenate(temp_y, axis=-1)])

    print(f"Dataset is: {len(dataset)} * {dataset[0].shape}")
    return dataset


if __name__ == '__main__':
    pass
