# ======================================================================
from learn import *
from vars import *
from model_ import *
import os
import cv2
import random
import glob


# ======================================================================


# ======================================================================

def get_dirs(dir_file_txt):
    """
    получает список директорий из текстового файла, удаляет символы "[" и "]" 
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
    генерирует случайный индекс для выбора случайного изображения из списка директорий.
    """
    rand_index = list(range(len(dir)))
    np.random.shuffle(rand_index)
    return rand_index


def read_img(dir):
    """
    читает изображение из директории, обрезает его, преобразует в черно-белое, изменяет 
    размер и возвращает его в виде массива numpy.
    """

    try:
        img = cv2.imread(dir, 0)[y1:y2, x1:x2].astype(np.bool_).astype(np.int8)
    except:
        print('read_img(dir)=', dir)
    img[:960] = -1
    img[img == 0] = -1
    img = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
    return np.expand_dims(img, axis=2)


def read_img25(dir):
    """
    читает изображение из директории, обрезает его, преобразует в черно-белое, 
    изменяет размер и возвращает его в виде массива numpy.
    """
    try:
        img = cv2.imread(dir, 0)[y1:y2, x1:x2].astype(np.bool_).astype(np.int8)
    except:
        print('read_img25(dir)=', dir)
    img[:960] = 0
    img = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
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
    img[:960] = 0
    img = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
    return img


def get_list_dir(root, dir):
    """
    возвращает список директорий для заданного файла и ракурсов.
    """
    # d = os.path.join(root, 'output_part' + dir + '_segmap.png')
    d = os.path.join(root, dir + '_segmap.png')

    i = dir.find('/', -3)

    # d_25 = os.path.join(root, 'output_part' + dir[:i+1], '25_segmap.png')
    d_25 = os.path.join(root, dir[:i + 1], '49_segmap.png')

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
    # список для хранения найденных путей к файлам
    file_paths = []

    # рекурсивно обойти все подкаталоги и найти файлы с заданным именем
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename == file_name:
                # если имя файла совпадает, добавить путь к файлу в список
                file_paths.append(os.path.join(dirpath, filename))

    # вернуть список найденных путей к файлам
    return file_paths


# ======================================================================

list_img_test, list_img_test_25 = [], []
dir = get_dirs(dir_file_txt)
dir_260_clear = get_dirs(dir_260_clear_file_txt)

# dir_test = np.load('data_a/dir_test.npy')
dir_test = np.array(random.choices(find_files_by_name(img_test, '9_segmap.png'), k=6))
# dir_test = ['/content/drive/MyDrive/Ступни/8channal_77/ready_renders/output_part1/yaw_0/3145left.pos/12/9_segmap.png',
#             '/content/drive/MyDrive/Ступни/8channal_77/ready_renders/output_part1/yaw_0/3091right.pos/12/9_segmap.png',
#             '/content/drive/MyDrive/Ступни/8channal_77/ready_renders/output_part1/yaw_0/3200right.pos/12/9_segmap.png',
#             '/content/drive/MyDrive/Ступни/8channal_77/ready_renders/output_part1/yaw_0/287left.pos/12/9_segmap.png',
#             '/content/drive/MyDrive/Ступни/8channal_77/ready_renders/output_part1/yaw_0/3122right.pos/12/9_segmap.png',
#             '/content/drive/MyDrive/Ступни/8channal_77/ready_renders/output_part1/yaw_0/3057left.pos/12/9_segmap.png', ]
print(f'type = {type(dir_test)}\n{dir_test}')
for d in dir_260_clear:
    name_model = get_name_model(d)
    if name_model not in list_models:
        list_models.append(name_model)

print('len(list_models)=', len(list_models))

for d in dir_test:
    list_img_test.append(np.concatenate(list(map(read_img, ((get_list_dir(img_test, d[24:-11]))[0]))), axis=-1))
    list_img_test_25.append(np.concatenate(list(map(read_img25, ((get_list_dir(img_test, d[24:-11]))[1]))), axis=-1))
list_img_test_array = np.array(list_img_test)

print('list_img_test shape=', list_img_test[0].shape, 'list_img_test 25 shape=', list_img_test_25[0].shape)

# ======================================================================

if __name__ == '__main__':
    pass
