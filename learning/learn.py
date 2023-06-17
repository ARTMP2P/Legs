import os.path

import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageChops
from torch.utils.data import DataLoader

# ======================================================================
from .vars import *
from .file_load import *
from .model_ import *


# ======================================================================


def summarize_performance(step, generator, dataset_list, device, save_model=True):
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

        inputs, labels = create_dataset(dataset_list, batch_size=1)
        inputs = torch.tensor(inputs).to(device).float().view(-1, 8, 1024, 1024)

        outputs = generator(inputs)

        generated_img = outputs.detach().cpu().numpy()[0]
        generated_img = np.where(generated_img >= 0, 1, generated_img)
        generated_img = np.where(generated_img < 0, 0, generated_img)
        generated_img = generated_img.astype(np.uint8)

        original_img = labels[0]

        for c in range(generated_img.shape[0]):
            generated_channel = generated_img[c, :, :]
            original_channel = original_img[:, :, c]

            difference = np.abs(generated_channel - original_channel)
            percentage = (np.count_nonzero(difference) * 100) / original_channel.size
            percentage_list.append(percentage)

            # Save the image with difference
            img_diff = np.concatenate(
                (
                    np.expand_dims(generated_channel * 255, 2),
                    np.expand_dims(original_channel * 255, 2),
                    np.zeros((1024, 1024, 1))
                ),
                axis=-1
            )
            img_diff = np.uint8(img_diff)

            cv2.imwrite(os.path.join(img_test_group, f'channel_{c + 1}.jpg'), img_diff)

        mean_percentage_diff = np.mean(percentage_list)
        print(f"Mean percentage difference for step {step}: {round(mean_percentage_diff, 2)}")

    generator.train()

    if save_model:
        if mean_percentage_diff <= 10:  # Set your desired threshold for saving the model
            filename_model = os.path.join(dir_model_NN, f'M_good{step}.pt')
        else:
            filename_model = os.path.join(dir_model_NN, f'M_{step}.pt')
        torch.save(generator.state_dict(), filename_model)
        print(f"> Saved model: {filename_model}")


def train(generator, discriminator, root_dir, num_epochs, batch_size, device):
    # Определение оптимизаторов и функции потерь
    generator_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    dataset_list = get_file_paths(root_dir)

    for epoch in range(num_epochs):
        batch_x, batch_y = create_dataset(dataset_list, batch_size)

        # Преобразование списков в один numpy.ndarray
        batch_x = np.array(batch_x)
        batch_y = np.array(batch_y)

        # Передача данных на устройство (GPU или CPU)
        batch_x = torch.tensor(batch_x).to(device)
        batch_y = torch.tensor(batch_y).to(device)

        # Изменение формы входного тензора
        batch_x = batch_x.view(-1, 8, 1024, 1024)
        batch_y = batch_y.view(-1, 8, 1024, 1024)

        # Обновление параметров дискриминатора
        discriminator_optimizer.zero_grad()

        # Прямой проход через дискриминатор
        real_labels = torch.ones(batch_size, 1, 125, 125).to(device)
        fake_labels = torch.zeros(batch_size, 1, 125, 125).to(device)
        real_outputs = discriminator(batch_y.float())
        fake_outputs = discriminator(generator(batch_x.float()))

        # Вычисление функции потерь для дискриминатора
        real_loss = criterion(real_outputs, real_labels)
        fake_loss = criterion(fake_outputs, fake_labels)
        discriminator_loss = real_loss + fake_loss

        # Обратное распространение и обновление параметров дискриминатора
        discriminator_loss.backward()
        discriminator_optimizer.step()

        # Обновление параметров генератора
        generator_optimizer.zero_grad()

        # Прямой проход через генератор
        outputs = generator(batch_x.float())
        generated_outputs = discriminator(outputs)

        # Вычисление функции потерь для генератора
        generator_loss = criterion(generated_outputs, real_labels)

        # Обратное распространение и обновление параметров генератора
        generator_loss.backward()
        generator_optimizer.step()

        # Вывод показателей потерь
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"Discriminator Loss: {discriminator_loss.item():.4f}")
        print(f"Generator Loss: {generator_loss.item():.4f}")

        # Проверка работы нейросети после каждой эпохи
        summarize_performance(epoch, generator, dataset_list, device)


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
