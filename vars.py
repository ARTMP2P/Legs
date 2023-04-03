CHANEL, SIZE = 8, 1024
n_steps, batch = 200000, 4
y1, y2, x1, x2 = 800, 2848, 136, 2184
lr = 0.0001
n_epochs=1


dir_file_txt = 'data_a/dir_260_clear_yaw_0.txt'
dir_260_clear_file_txt = 'data_a/dir_260_clear.txt'
dir_model_NN = 'data_a/models'
img_test_group = 'data_a/img_test_group'
img_test = 'data_a/img_test'
root_25 = 'data_a/rendered_models'
root = 'data_a/ready_renders/'



rakurs = ['yaw_0',
          'yaw_55',
          'yaw_90',
          'yaw_125',
          'yaw_180',
          'yaw_235',
          'yaw_270',
          'yaw_305']


'''
Zeros data
'''
dir_260_clear = []
list_img_test, list_img_test_25 = [], []
list_models = []
bugs = []
dir_260_clear = []
