import torch
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


def summarize_performance(step, g_model, f=0):
    """
    This function is used to save trained models and evaluate their performance on the test data.
    The function takes as arguments the training step number, the trained model, and the flag f that controls
    the model file name when saving. If f is equal to 1, the file name will include an additional 'good' label,
    and if f is equal to 0, the 'good' label is not added.

    The function uses the saved model to generate images based on the test data. Then, it compares the generated
    images with the original images from the test data and computes the percentage difference between them.
    The results are saved as images in the 'img_test' folder and also printed on the screen as the average
    percentage difference for each test image.
    """
    if f:
        filename_model_NN = f'{dir_model_NN}/M_good{step}.pt'
        torch.save(g_model.state_dict(), filename_model_NN)
        print('> Saved:', filename_model_NN)
    else:
        filename_model_NN = f'{dir_model_NN}/M_{step}.pt'
        torch.save(g_model.state_dict(), filename_model_NN)
        print('> Saved:', filename_model_NN)

    try:
        X = g_model(torch.tensor(list_img_test)).detach().numpy()
    except:
        print('Error!')

    for j, im in enumerate(X):
        precentage_list = []

        for i in range(CHANEL):
            IMG = np.concatenate((np.expand_dims(im[:, :, i] * 255, 2),
                                   np.expand_dims(list_img_test_25[j][:, :, i] * 255, 2),
                                   np.zeros((SIZE, SIZE, 1))), axis=-1)

            differ = cv2.absdiff(list_img_test_25[j][:, :, i].astype(np.float64), im[:, :, i].astype(np.float64))
            differ = differ.astype(np.uint8)
            percentage = (np.count_nonzero(differ) * 100) / differ.size
            precentage_list.append(percentage)

            IMG_res = cv2.resize(IMG, (int(SIZE * 2), int(SIZE * 2)), interpolation=cv2.INTER_NEAREST)
            IMG_res = IMG_res[:, :, ::-1]
            cv2.imwrite(f'{img_test_group}/{rakurs[i]}_{dir_test[j][75:-20]}{j}.jpg', np.uint8(IMG_res))
            print(f"Percentage for {rakurs[i]} is: {round(percentage, 2)}")
        print(f"Mean percentage for model {j} is: {round(np.mean(precentage_list), 2)}")
    print(f"Mean percentage for all models is: {round(np.mean(precentage_list), 2)}")
    with open(log_file, 'a+') as file:
        file.write(f'{filename_model_NN}\nMetricks: {np.mean(precentage_list)}\n')


def train(d_model, dir, n_epochs=200, n_batch=1, i_s=0, bufer=0):
    """
    This function is used to train a Generative Adversarial Network (GAN) model for style transfer between images.
    It takes as input the discriminator model (d_model), generator model (g_model), and GAN model (gan_model),
    the number of epochs (n_epochs), batch size (n_batch), start epoch index (i_s), and buffer (bufer).

    Inside the function, it generates real and fake images, updates the parameters of the discriminator and generator,
    and computes and prints the training results. The function also calls auxiliary functions to display statistics
    and training results.
    """
    shown_statics()
    i_s = 0
    # determine the output square shape of the discriminator
    n_patch = d_model.output_shape[1]
    print(f'n_patch {n_patch}\nn_epochs = {n_epochs}')

    for i in range(n_steps):  # n_steps

        list_A, list_B, list_y = [], [], []
        list_rand_dir, list_rand_dir_25 = get_list_dir_2(root, list_models, batch)
        # print(list_rand_dir, '\n', list_rand_dir_25)

        for i_d in range(batch):
            list_dir_name, list_dir_name_25 = list_rand_dir[i_d], list_rand_dir_25[i_d]

            X_A, X_B, y = generate_real_samples(list_dir_name, list_dir_name_25, n_patch)
            list_A.append(X_A)
            list_B.append(X_B)
            list_y.append(y)

        X_realA = np.concatenate(list_A, axis=0)
        X_realB = np.concatenate(list_B, axis=0)
        y_real = np.concatenate(list_y, axis=0)

        X_realA = torch.tensor(X_realA).float()
        X_realB = torch.tensor(X_realB).float()
        y_real = torch.tensor(y_real).float()

        # Generate fake samples
        X_fakeB, y_fake = generate_fake_samples(g_model, X_realA, n_patch)

        X_fakeB = torch.tensor(X_fakeB).float()
        y_fake = torch.tensor(y_fake).float()

        # Train the discriminator
        d_loss1 = d_model.train_on_batch([X_realA, X_realB], y_real)
        d_loss2 = d_model.train_on_batch([X_realA, X_fakeB], y_fake)

        # Train the generator
        g_loss, _, _ = gan_model.train_on_batch(X_realA, [y_real, X_realB])

        # Summarize performance
        print('>%d, d1[%.3f] d2[%.3f] g[%.3f]' % (i + 1, d_loss1, d_loss2, g_loss))

        # Summarize model performance
        if (i + 1) % n_epochs == 0:
            bufer = i
            i_s += 1
            summarize_performance(i_s, g_model)
            # break

        if (g_loss < 0.90) and (i - bufer > 25):
            bufer = i
            i_s += 1
            summarize_performance(i_s, g_model, f=1)
            # break


def train_on_dataset(model, dataset, epochs, optimizer, batch_size):
    # Create DataLoader to iterate through dataset
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Iterate over given number of epochs
    for epoch in range(epochs):
        running_loss = 0.0

        # Iterate over data loader
        for i, sample in enumerate(data_loader):
            # Initialize gradients
            optimizer.zero_grad()

            # Get image and label
            image, label = sample

            # Pass image through model
            output = model(image)

            # Calculate loss
            loss = criterion(output, label)

            # Run backpropagation
            loss.backward()

            # Update model parameters
            optimizer.step()

            # Update running loss
            running_loss += loss.item()

            # Print logs after every 1000 iterations
            if i % 1000 == 999:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 1000))
                running_loss = 0.0

    # Return trained model
    return model


def shown_statics():
    print(f'Lerning rate: {lr}\nBatch: {batch}')


if __name__ == '__main__':
    pass
