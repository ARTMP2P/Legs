import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageChops
from statistics import mean

# ======================================================================
from .vars import *
from .file_load import *
from .model_ import *


# ======================================================================


def summarize_performance(step, generator, dataloader, dataset_name, device, save_model=True):
    """
    This function is used to save trained models and evaluate their performance on the test data.
    The function takes as arguments the training step number, the trained generator model, the dataloader
    for the test data, the name of the dataset, and the device (e.g., 'cuda' or 'cpu').

    The function uses the generator model to generate images based on the test data. Then, it compares the generated
    images with the original images from the test data and computes the percentage difference between them.
    The results are saved as images in the 'img_test' folder and also printed on the screen as the average
    percentage difference for each test image.
    """
    generator.eval()
    with torch.no_grad():
        percentage_list = []

        for _, data in enumerate(dataloader):
            inputs = data["input"].to(device)
            labels = data["label"].to(device)

            outputs = generator(inputs)
            batch_size = inputs.size(0)

            for i in range(batch_size):
                generated_img = outputs[i].detach().cpu().numpy()
                original_img = labels[i].detach().cpu().numpy()

                for c in range(generated_img.shape[2]):
                    generated_channel = generated_img[:, :, c]
                    original_channel = original_img[:, :, c]
                    difference = np.abs(generated_channel - original_channel)
                    percentage = (np.count_nonzero(difference) * 100) / original_channel.size
                    percentage_list.append(percentage)

                    # Save the image with difference
                    img_diff = np.concatenate((np.expand_dims(generated_channel * 255, 2),
                                               np.expand_dims(original_channel * 255, 2),
                                               np.zeros((1024, 1024, 1))), axis=-1)
                    img_diff = np.uint8(img_diff)
                    cv2.imwrite(f'img_test/{dataset_name}_{step}_channel{c}.jpg', img_diff)

                    # Print the percentage difference
                    print(f"Mean percentage difference for image {_}, channel {c}: {round(percentage, 2)}")

        mean_percentage_diff = np.mean(percentage_list)
        print(f"Mean percentage difference for dataset {dataset_name}, step {step}: {round(mean_percentage_diff, 2)}")

    generator.train()

    if save_model:
        if mean_percentage_diff <= 10:  # Set your desired threshold for saving the model
            filename_model = f'models/M_good{step}.pt'
        else:
            filename_model = f'models/M_{step}.pt'
        torch.save(generator.state_dict(), filename_model)
        print(f"> Saved model: {filename_model}")


def train(generator, discriminator, dataset_path, batch_size, num_epochs):
    # Определение функции потерь и оптимизаторов
    # Загрузка и предобработка датасета
    dataset = torch.load(dataset_path)
    # Дополнительная предобработка датасета (нормализация, аугментация и т.д.)

    # Создание DataLoader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # Определение функции потерь и оптимизатора
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(generator.parameters(), lr=0.001)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator.to(device)
    discriminator.to(device)

    # Цикл обучения
    for epoch in range(num_epochs):
        generator.train()

        for i, data in enumerate(dataloader):
            # Обработка входных данных и меток
            inputs = data[:-1].to(device)
            labels = data[-1].to(device)

            # Обнуление градиентов
            optimizer.zero_grad()

            # Прямой проход генератора
            outputs = generator(inputs)

            # Вычисление функции потерь и обратный проход
            loss = criterion(outputs, labels)
            loss.backward()

            # Обновление весов генератора
            optimizer.step()

        # Сохранение модели и оценка производительности
        summarize_performance(epoch, generator, dataloader, "dataset_name", device)

    print("Training completed.")


# def summarize_performance(step, generator, dataloader, f=0):
#     """
#     This function is used to save trained models and evaluate their performance on the test data.
#     The function takes as arguments the training step number, the trained generator model, and the flag f that controls
#     the model file name when saving. If f is equal to 1, the file name will include an additional 'good' label,
#     and if f is equal to 0, the 'good' label is not added.
#
#     The function uses the generator model to generate images based on the test data. Then, it compares the generated
#     images with the original images from the test data and computes the percentage difference between them.
#     The results are saved as images in the 'img_test' folder and also printed on the screen as the average
#     percentage difference for each test image.
#     """
#     if f:
#         filename_model_NN = f'{dir_model_NN}/M_good{step}.pt'
#         torch.save(generator.state_dict(), filename_model_NN)
#         print('> Saved:', filename_model_NN)
#     else:
#         filename_model_NN = f'{dir_model_NN}/M_{step}.pt'
#         torch.save(generator.state_dict(), filename_model_NN)
#         print('> Saved:', filename_model_NN)
#
#     try:
#         generator.eval()
#         with torch.no_grad():
#             for _, (inputs, labels) in enumerate(dataloader):
#                 outputs = generator(inputs)
#                 percentage_list = []
#
#                 for i in range(len(outputs)):
#                     generated_img = outputs[i].detach().numpy()
#                     original_img = labels[i].detach().numpy()
#
#                     for c in range(generated_img.shape[2]):
#                         generated_channel = generated_img[:, :, c]
#                         original_channel = original_img[:, :, c]
#                         difference = np.abs(generated_channel - original_channel)
#                         percentage = (np.count_nonzero(difference) * 100) / original_channel.size
#                         percentage_list.append(percentage)
#
#                         # Сохраняем изображение с разницей
#                         img_diff = np.concatenate((np.expand_dims(generated_channel * 255, 2),
#                                                    np.expand_dims(original_channel * 255, 2),
#                                                    np.zeros((SIZE, SIZE, 1))), axis=-1)
#                         #img_diff = np.concatenate((generated_channel * 255, original_channel * 255, difference * 255),
#                         #                          axis=-1)
#                         #img_diff_resized = cv2.resize(img_diff, (int(SIZE), int(SIZE)), interpolation=cv2.INTER_NEAREST)
#                         # img_diff_resized = img_diff_resized[:, :, ::-1]
#                         cv2.imwrite(f'{img_test_group}/{dir_test[75:-20]}_channel{c}.jpg',
#                                     np.uint8(img_diff))
#
#                         # print(f"Percentage difference for image {j}-{i}, channel {c}: {round(percentage, 2)}")
#
#                 print(f"Mean percentage difference for image {_}: {round(np.mean(percentage_list), 2)}")
#
#         generator.train()
#     except Exception as e:
#         print('Error:', str(e), img_diff.shape, generated_img.shape, original_img.shape)


# def train(generator, dataset, num_epochs, batch_size, patch_shape):
#     criterion = nn.MSELoss()
#     optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
#
#     dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
#
#     for epoch in range(num_epochs):
#         epoch_loss = 0.0  # Переменная для накопления значения ошибки в текущей эпохе
#         num_batches = 0  # Переменная для подсчета количества пакетов в текущей эпохе
#
#         for batch_idx, (inputs, labels) in enumerate(dataloader):
#             optimizer.zero_grad()
#             outputs = generator(inputs)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#
#             epoch_loss += loss.item()
#             num_batches += 1
#
#             if (batch_idx + 1) % 10 == 0:
#                 print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, batch_idx + 1,
#                                                                          len(dataloader), loss.item()))
#
#         average_loss = epoch_loss / num_batches
#         print('Epoch [{}/{}], Average Loss: {:.4f}'.format(epoch + 1, num_epochs, average_loss))
#
#         # Проверка производительности после каждой эпохи
#         summarize_performance(epoch + 1, generator, dataloader, f=1)


def shown_statics():
    print(f'Lerning rate: {lr}\nBatch: {batch}')


if __name__ == '__main__':
    pass
