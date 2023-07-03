# ======================================================================
import argparse

from learning.file_load import *
from learning.learn import *
from learning.model_ import *
import parser

# ======================================================================


# n_epochs = 20
image_shape = [SIZE, SIZE, CHANEL]
d_model = define_discriminator(image_shape)
parser = argparse.ArgumentParser()
parser.add_argument("file_path", nargs="?",
                    default=None, help="Путь к обученной модели (необязательно для дообучения)")
parser.add_argument("epochs", type=int,
                    default=200, help="Количество эпох обучения")
parser.add_argument("side", type=str,
                    default="Right", choices=["Left", "Right"], help="Задает сторону (Left или Right)")

args = parser.parse_args()

n_epochs = args.epochs

if args.file_path:
    g_model = load_model(args.file_path)
    print(f'Learning running from {args.file_path}')
else:
    g_model = define_generator(image_shape)
    print('Learning running with start')
gan_model = define_gan(g_model, d_model, image_shape)
train(d_model, g_model, gan_model, dir, n_epochs, args.side)

# train(d_model, g_model, gan_model, dataset, n_epochs)   /content/drive/MyDrive/Ступни/8channal_77/model/M_1.h5


# ======================================================================

if __name__ == '__main__':
    # gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    # for device in gpu_devices:
    #     tf.config.experimental.set_memory_growth(device, True)
    print(get_file_paths(root, "Left"))
