# ======================================================================
from .vars import *
from .model_ import *
import os
import cv2
import random
import glob


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
    """
    # try:
    #     img = cv2.imread(dir, 0)[y1:y2, x1:x2].astype(np.bool_).astype(np.int8)
    # except:
    #     print('read_img25(dir)=', dir)
    # # img[:960] = 0
    # img = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
    img = cv2.imread(dir, 0).astype(np.bool_).astype(np.int8)
    return np.expand_dims(img, axis=2)


def get_img(dir):
    """
    читает изображение из директории, обрезает его, преобразует в черно-белое, 
    изменяет размер и возвращает его в виде массива numpy.
    """
    try:
        img = cv2.imread(dir, 0)[y1:y2, x1:x2].astype(np.bool_).astype(np.int8)
    except:
        print('get_img(dir)=', dir)
    # img[:960] = 0
    img = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
    return img


def get_list_dir(dir):
    """
    Возвращает список директорий для заданного файла и ракурсов.
    """
    # d = os.path.join(root, 'output_part' + dir + '_segmap.png')
    print(f"Dir is: {dir}, type is: {type(dir)}")
    d = os.path.join(dir[0][:-len('_segmap.png')] + '_segmap.png')

    # d_25 = os.path.join(root, 'output_part' + dir[:i+1], '25_segmap.png')
    d_25 = os.path.join(dir[0][:-len('_segmap.png')], '49_segmap.png')

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
    with open(log_file, 'a+') as file:
        file.write(f'list_rand_dir: {list_rand_dir}\nlist_rand_dir_25: {list_rand_dir_25}\n')
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

list_img_test, list_img_test_25 = [], []
dir = get_dirs(dir_file_txt)
dir_260_clear = get_dirs(dir_260_clear_file_txt)

dir_test = np.array(find_files_by_name(img_test, '9_segmap.png'))
print(dir_test.shape)

for d in dir_260_clear:
    name_model = get_name_model(d)
    if name_model not in list_models:
        list_models.append(name_model)

for d in dir_test:
  list_img_test.append(np.concatenate(list(map(read_img, ((get_list_dir([d]))[0]))), axis=-1))
  list_img_test_25.append(np.concatenate(list(map(read_img25, ((get_list_dir([d]))[1]))), axis=-1))

print('list_img_test shape=', list_img_test[0].shape, 'list_img_test 25 shape=', list_img_test_25[0].shape)

# ======================================================================

if __name__ == '__main__':
    pass
