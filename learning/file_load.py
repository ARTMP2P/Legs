# ======================================================================
from .vars import *
from .model_ import *
import os
import cv2
import random
import glob
import tqdm


# ======================================================================


# ======================================================================

def get_file_paths(root_dir: str, side: str) -> list:
    """
    Возвращает список путей к файлам, содержащим заданное значение `side` в пути.

    :param root_dir: Корневая директория, в которой производится поиск.
    :param side: Значение, которое должно содержаться в пути.
    :return: Список путей к файлам.
    """
    subfolder_list = sorted(os.listdir(root_dir))  # Список подпапок с номерами моделей
    file_paths = []

    for subfolder_model in subfolder_list:
        model_dir = os.path.join(root_dir, subfolder_model)

        for dirpath, _, filenames in os.walk(model_dir):
            for filename in filenames:
                if filename == "0_segmap.png":
                    if "yaw_0" in dirpath and os.path.exists(os.path.join(dirpath, '0_segmap.png')):
                        if side in dirpath:  # Добавленная проверка на наличие `side` в пути
                            file_paths.append(dirpath)

    return file_paths


def create_dataset(file_paths: list, batch_size: int) -> list:
    """

    :param file_paths:
    :param batch_size:
    :return:
    """
    left_yaw_list = []
    right_yaw_list = []
    batch_x, batch_y = [], []
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

        batch_x.append(np.concatenate(temp_x, axis=-1))

        for m in ['0', '55', '90', '125', '180', '235', '270', '305']:
            t_file_path = file_path.replace('yaw_0', f"yaw_{m}")
            true_file_path = os.path.join(t_file_path, f"49_segmap.png")

            try:
                true_tensor = read_img(true_file_path)
                temp_y.append(true_tensor)
            except:
                print(true_file_path)

        batch_y.append(np.concatenate(temp_y, axis=-1))

    # print(f"Dataset is: {len(dataset)}")
    return batch_x, batch_y
# ======================================================================

if __name__ == '__main__':
    pass
