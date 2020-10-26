'''
This module will attempt to predict model parameters by using a trained model.
'''

import tensorflow as tf
import os
import h5py
import numpy as np
import pickle

base_dir = os.path.expanduser('~/fluoro/data/compilation')

hist_path = os.path.expanduser('~/fluoro/code/jupyt/vox_fluoro/vox_fluoro_img_stnd_loss')
hist_file_name = 'vox_fluoro_hist_objects_2.pkl'
load_model_name = 'vox_fluoro_img_stnd_loss_2.h5'

save_dir = os.path.abspath(os.path.expanduser('~/fluoro/code/jupyt/update_2019-Sep-03'))
save_file_name = 'vox_fluoro_predict_L1_0-1_L2_0-1_dist.pkl'

predict_numb = 100



def cust_mean_squared_error_std(y_true, y_pred):
    base_dir = os.path.expanduser('~/fluoro/data/compilation')
    stats_file = h5py.File(os.path.join(base_dir, 'labels_stats.h5py'), 'r')
    # mean_dset = stats_file['mean']
    std_dset = stats_file['std']
    # var_dset = stats_file['var']
    # mean_v = mean_dset[:]
    std_v = std_dset[:]
    # var_v = var_dset[:]

    stats_file.close()


    return tf.keras.backend.mean(tf.keras.backend.square((y_pred - y_true) / std_v))




hist_file = open(os.path.join(hist_path, hist_file_name), 'rb')
hist_data = pickle.load(hist_file)

hist_file.close()


os.makedirs(save_dir, exist_ok=True)

random_test_values = np.random.choice(hist_data['test_indxs'], size=predict_numb, replace=False)

vox_file = h5py.File(os.path.expanduser('~/fluoro/data/compilation/voxels_pad.h5py'), 'r')
vox_init = vox_file['vox_dset']
vox_test_mat = vox_init[sorted(random_test_values)]
vox_file.close()

image_file = h5py.File(os.path.expanduser('~/fluoro/data/compilation/images.h5py'), 'r')
image_init = image_file['image_dset']
image_test_mat = image_init[sorted(random_test_values)]
image_file.close()

label_file = h5py.File(os.path.expanduser('~/fluoro/data/compilation/labels.h5py'), 'r')
label_init = label_file['labels_dset']
label_test_mat = label_init[sorted(random_test_values)]
label_file.close()

cali_file = h5py.File(os.path.expanduser('~/fluoro/data/compilation/calibration.h5py'), 'r')
cali_init = cali_file['cali_len3_rot']
cali_test_mat = cali_init[sorted(random_test_values)]
cali_file.close()

model = tf.keras.models.load_model(os.path.join(hist_path, load_model_name), custom_objects={'cust_mean_squared_error_std': cust_mean_squared_error_std})



predict_1 = model.predict([np.expand_dims(vox_test_mat, axis=-1), np.expand_dims(image_test_mat[:, 0, :, :], axis=-1), np.expand_dims(image_test_mat[:, 1, :, :], axis=-1), cali_test_mat], batch_size=10, verbose=1)

save_file = open(os.path.join(save_dir, save_file_name), 'wb')

output_dict = {}

output_dict['predictions'] = predict_1
output_dict['actual'] = label_test_mat

pickle.dump(output_dict, save_file)

save_file.close()
