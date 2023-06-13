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


# def summarize_performance(step, g_model, f=0):
#     """
#     This function is used to save trained models and evaluate their performance on the test data.
#     The function takes as arguments the training step number, the trained model, and the flag f that controls
#     the model file name when saving. If f is equal to 1, the file name will include an additional 'good' label,
#     and if f is equal to 0, the 'good' label is not added.
#
#     The function uses the saved model to generate images based on the test data. Then, it compares the generated
#     images with the original images from the test data and computes the percentage difference between them.
#     The results are saved as images in the 'img_test' folder and also printed on the screen as the average
#     percentage difference for each test image.
#     """
#     if f:
#         filename_model_NN = f'{dir_model_NN}/M_good{step}.pt'
#         torch.save(g_model.state_dict(), filename_model_NN)
#         print('> Saved:', filename_model_NN)
#     else:
#         filename_model_NN = f'{dir_model_NN}/M_{step}.pt'
#         torch.save(g_model.state_dict(), filename_model_NN)
#         print('> Saved:', filename_model_NN)
#
#     try:
#         X = g_model(torch.tensor(list_img_test)).detach().numpy()
#     except:
#         print('Error!')
#
#     for j, im in enumerate(X):
#         precentage_list = []
#
#         for i in range(CHANEL):
#             IMG = np.concatenate((np.expand_dims(im[:, :, i] * 255, 2),
#                                    np.expand_dims(list_img_test_25[j][:, :, i] * 255, 2),
#                                    np.zeros((SIZE, SIZE, 1))), axis=-1)
#
#             differ = cv2.absdiff(list_img_test_25[j][:, :, i].astype(np.float64), im[:, :, i].astype(np.float64))
#             differ = differ.astype(np.uint8)
#             percentage = (np.count_nonzero(differ) * 100) / differ.size
#             precentage_list.append(percentage)
#
#             IMG_res = cv2.resize(IMG, (int(SIZE * 2), int(SIZE * 2)), interpolation=cv2.INTER_NEAREST)
#             IMG_res = IMG_res[:, :, ::-1]
#             cv2.imwrite(f'{img_test_group}/{rakurs[i]}_{dir_test[j][75:-20]}{j}.jpg', np.uint8(IMG_res))
#             print(f"Percentage for {rakurs[i]} is: {round(percentage, 2)}")
#         print(f"Mean percentage for model {j} is: {round(np.mean(precentage_list), 2)}")
#     print(f"Mean percentage for all models is: {round(np.mean(precentage_list), 2)}")
#     with open(log_file, 'a+') as file:
#         file.write(f'{filename_model_NN}\nMetricks: {np.mean(precentage_list)}\n')


# def train0(d_model, g_model, gan_model, n_epochs=200, n_batch=1, i_s=0, bufer=0):
#     shown_statics()
#     i_s = 0
#     n_patch = d_model.conv5.out_channels  # Определение размерности выходной карты признаков дискриминатора
#     print(f'n_patch {n_patch}\nn_epochs = {n_epochs}')
#
#     # Оптимизаторы для дискриминатора и генератора
#     d_optimizer = optim.Adam(d_model.parameters(), lr=0.0002, betas=(0.5, 0.999))
#     g_optimizer = optim.Adam(g_model.parameters(), lr=0.0002, betas=(0.5, 0.999))
#
#     for i in range(n_steps):
#         list_A, list_B, list_y = [], [], []
#         list_rand_dir, list_rand_dir_25 = get_list_dir_2(root, list_models, batch)
#
#         for i_d in range(batch):
#             list_dir_name, list_dir_name_25 = list_rand_dir[i_d], list_rand_dir_25[i_d]
#
#             [X_A, X_B], y = generate_real_samples(list_dir_name, list_dir_name_25, n_patch, n_batch)
#             list_A.append(X_A)
#             list_B.append(X_B)
#             list_y.append(y)
#
#         X_realA = np.concatenate(list_A, axis=0)
#         X_realA = np.transpose(X_realA, (0, 3, 1, 2))
#         X_realB = np.concatenate(list_B, axis=0)
#         X_realB = np.transpose(X_realB, (0, 3, 1, 2))
#         y_real = np.concatenate(list_y, axis=0)
#
#         X_realA = torch.tensor(X_realA).float()
#         X_realB = torch.tensor(X_realB).float()
#         y_real = torch.tensor(y_real).float()
#         print(f"X_realA {X_realA.shape}\nX_realB {X_realB.shape}\ny_real {y_real.shape}")
#
#         # Generate fake samples
#         X_fakeB, y_fake = generate_fake_samples(g_model, X_realA, n_batch, n_patch)
#
#         X_fakeB = torch.tensor(X_fakeB).float()
#         y_fake = torch.tensor(y_fake).float()
#         print(f"X_fakeB {X_fakeB.shape}\ny_fake {y_fake.shape}")
#
#         # Train the discriminator
#         d_loss1 = torch.mean(d_model(X_realA, X_realB))
#         d_loss2 = torch.mean(d_model(X_realA, X_fakeB))
#
#         # Обновить параметры дискриминатора
#         d_optimizer.zero_grad()
#         d_loss = d_loss1 + d_loss2
#         d_loss.backward()
#         d_optimizer.step()
#
#         # Train the generator
#         g_loss = gan_model(y_real)  # Вычислить потерю генератора
#
#         # Обновить параметры генератора
#         g_optimizer.zero_grad()
#         g_loss.backward()
#         g_optimizer.step()
#
#         # Summarize performance
#         print('>%d, d1[%.3f] d2[%.3f] g[%.3f]' % (i + 1, d_loss1.item(), d_loss2.item(), g_loss.item()))
#
#         # Summarize model performance
#         if (i + 1) % n_epochs == 0:
#             bufer = i
#             i_s += 1
#             summarize_performance(i_s, g_model)
#             # break
#
#         if (g_loss < 0.90) and (i - bufer > 25):
#             bufer = i
#             i_s += 1
#             summarize_performance(i_s, g_model, f=1)
#             # break
def summarize_performance(step, generator, dataloader, f=0):
    """
    This function is used to save trained models and evaluate their performance on the test data.
    The function takes as arguments the training step number, the trained generator model, and the flag f that controls
    the model file name when saving. If f is equal to 1, the file name will include an additional 'good' label,
    and if f is equal to 0, the 'good' label is not added.

    The function uses the generator model to generate images based on the test data. Then, it compares the generated
    images with the original images from the test data and computes the percentage difference between them.
    The results are saved as images in the 'img_test' folder and also printed on the screen as the average
    percentage difference for each test image.
    """
    if f:
        filename_model_NN = f'{dir_model_NN}/M_good{step}.pt'
        torch.save(generator.state_dict(), filename_model_NN)
        print('> Saved:', filename_model_NN)
    else:
        filename_model_NN = f'{dir_model_NN}/M_{step}.pt'
        torch.save(generator.state_dict(), filename_model_NN)
        print('> Saved:', filename_model_NN)

    try:
        generator.eval()
        with torch.no_grad():
            for j, (inputs, labels) in enumerate(dataloader):
                inputs = inputs.permute(0, 3, 1, 2)

                outputs = generator(inputs)
                percentage_list = []

                for i in range(len(outputs)):
                    generated_img = outputs[i].detach().numpy()
                    original_img = labels[i].detach().numpy()

                    for c in range(generated_img.shape[2]):
                        generated_channel = generated_img[:, :, c]
                        original_channel = original_img[:, :, c]
                        difference = np.abs(generated_channel - original_channel)
                        percentage = (np.count_nonzero(difference) * 100) / original_channel.size
                        percentage_list.append(percentage)

                        # Сохраняем изображение с разницей
                        img_diff = np.concatenate((generated_channel * 255, original_channel * 255, difference * 255),
                                                  axis=-1)
                        img_diff_resized = cv2.resize(img_diff, (int(SIZE), int(SIZE)), interpolation=cv2.INTER_NEAREST)
                        img_diff_resized = img_diff_resized[:, :, ::-1]
                        cv2.imwrite(f'{img_test_group}/{dir_test[j][75:-20]}{j}_channel{c}.jpg',
                                    np.uint8(img_diff_resized))

                        print(f"Percentage difference for image {j}-{i}, channel {c}: {round(percentage, 2)}")

                print(f"Mean percentage difference for image {j}: {round(np.mean(percentage_list), 2)}")

        generator.train()
    except Exception as e:
        print('Error:', str(e), img_diff_resized.shape, outputs.shape, inputs.shape)


def train(generator, dataset, num_epochs, batch_size, patch_shape):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        epoch_loss = 0.0  # Переменная для накопления значения ошибки в текущей эпохе
        num_batches = 0  # Переменная для подсчета количества пакетов в текущей эпохе

        for batch_idx, (inputs, labels) in enumerate(dataloader):
            optimizer.zero_grad()
            outputs = generator(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

            if (batch_idx + 1) % 10 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, batch_idx + 1,
                                                                         len(dataloader), loss.item()))

        average_loss = epoch_loss / num_batches
        print('Epoch [{}/{}], Average Loss: {:.4f}'.format(epoch + 1, num_epochs, average_loss))

        # Проверка производительности после каждой эпохи
        summarize_performance(epoch + 1, generator, dataloader, f=1)


def shown_statics():
    print(f'Lerning rate: {lr}\nBatch: {batch}')


if __name__ == '__main__':
    pass
