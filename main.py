import tensorflow as tf

# ======================================================================
from learn import *
from model_ import *

# ======================================================================



n_epochs = 1
image_shape = [SIZE, SIZE, CHANEL]
d_model = define_discriminator(image_shape)
g_model = define_generator(image_shape)
# g_model = load_model('Ступни/8channal_260/models/M_1936.h5')
gan_model = define_gan(g_model, d_model, image_shape)

# train(d_model, g_model, gan_model, dataset, n_epochs)   /content/drive/MyDrive/Ступни/8channal_77/model/M_1.h5
train(d_model, g_model, gan_model, dir, n_epochs)


# ======================================================================

if __name__ == '__main__':
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)
    
