import os
from keras.models import load_model

current_path = os.path.dirname(os.path.realpath(__file__))
RAKURSES = ['0', '35', '90', '145', '180', '215', '270', '325']
g_model = load_model(os.path.join(current_path, 'data_a', 'models', 'M_1.h5'), compile=True)
