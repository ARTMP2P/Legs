import os
from keras.models import load_model

current_path = os.getcwd()
g_model1 = load_model(os.path.join(current_path, 'data_a', 'models', 'M_100_Left.h5'), compile=False)
g_model2 = load_model(os.path.join(current_path, 'data_a', 'models', 'M_100_Right.h5'), compile=False)
