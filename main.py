# ======================================================================
import argparse

from learning.file_load import *
from learning.learn import *
from learning.model_ import *
import parser

# ======================================================================


# n_epochs = 20
image_shape = [batch, CHANEL, SIZE, SIZE]
d_model = Discriminator(image_shape)
parser = argparse.ArgumentParser()
parser.add_argument("file_path", nargs="?", default=None)
parser.add_argument("epochs", type=int, default=200)
args = parser.parse_args()

n_epochs = args.epochs

if args.file_path:
    g_model = torch.load(args.file_path)
    print(f'Learning running from {args.file_path}')
else:
    g_model = UNet(image_shape)
    print('Learning running with start')
gan_model = get_model(image_shape)
train(gan_model, n_epochs=10)

# train(d_model, g_model, gan_model, dataset, n_epochs)   /content/drive/MyDrive/Ступни/8channal_77/model/M_1.h5


# ======================================================================

if __name__ == '__main__':
    pass
    # gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    # for device in gpu_devices:
    #     tf.config.experimental.set_memory_growth(device, True)
