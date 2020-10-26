'''
This module will attempt to predict model parameters by using a trained model.
'''

import tensorflow as tf
import os
import h5py
import numpy as np
import pickle

base_dir = os.path.expanduser('~/fluoro/data/compilation')

file_base_name = 'vox_fluoro_res'

hist_path = os.path.join(os.path.expanduser('~/fluoro/code/jupyt/vox_fluoro'), file_base_name)
hist_file_name = file_base_name + '_hist_objects_1.pkl'
load_model_name = file_base_name + '_1.h5'

save_dir = os.path.abspath(os.path.expanduser('~/fluoro/code/jupyt/update_2019-Sep-17/predictions'))
os.makedirs(save_dir, exist_ok=True)

save_file_name = 'model_prediction_' + file_base_name + '.pkl'

predict_numb = 100

# -----------------------------------------------------------------


def cust_mean_squared_error_var(y_true, y_pred):
    base_dir = os.path.expanduser('~/fluoro/data/compilation')
    stats_file = h5py.File(os.path.join(base_dir, 'labels_stats.h5py'), 'r')
    # mean_dset = stats_file['mean']
    # std_dset = stats_file['std']
    var_dset = stats_file['var']
    # mean_v = mean_dset[:]
    # std_v = std_dset[:]
    var_v = var_dset[:]

    stats_file.close()


    return tf.keras.backend.mean(tf.keras.backend.square(y_pred - y_true) / var_v)

# -----------------------------------------------------------------


hist_file = open(os.path.join(hist_path, hist_file_name), 'rb')
hist_data = pickle.load(hist_file)

hist_file.close()

random_test_values = np.random.choice(hist_data['test_indxs'], size=predict_numb, replace=False)
random_train_values = np.random.choice(hist_data['train_indxs'], size=predict_numb, replace=False)

vox_file = h5py.File(os.path.expanduser('~/fluoro/data/compilation/voxels_pad.h5py'), 'r')
vox_init = vox_file['vox_dset']
vox_test_mat = vox_init[sorted(random_test_values)]
vox_train_mat = vox_init[sorted(random_train_values)]
vox_file.close()

image_file = h5py.File(os.path.expanduser('~/fluoro/data/compilation/images.h5py'), 'r')
image_init = image_file['image_dset']
image_test_mat = image_init[sorted(random_test_values)]
image_train_mat = image_init[sorted(random_train_values)]
image_file.close()

label_file = h5py.File(os.path.expanduser('~/fluoro/data/compilation/labels.h5py'), 'r')
label_init = label_file['labels_dset']
label_test_mat = label_init[sorted(random_test_values)]
label_train_mat = label_init[sorted(random_train_values)]
label_file.close()

cali_file = h5py.File(os.path.expanduser('~/fluoro/data/compilation/calibration.h5py'), 'r')
cali_init = cali_file['cali_len3_rot']
cali_test_mat = cali_init[sorted(random_test_values)]
cali_train_mat = cali_init[sorted(random_train_values)]
cali_file.close()

# -----------------------------------------------------------------

model = tf.keras.models.load_model(os.path.join(hist_path, load_model_name), custom_objects={'cust_mean_squared_error_var': cust_mean_squared_error_var})


predict_test = model.predict([np.expand_dims(vox_test_mat, axis=-1), np.expand_dims(image_test_mat[:, 0, :, :], axis=-1), np.expand_dims(image_test_mat[:, 1, :, :], axis=-1), cali_test_mat], batch_size=10, verbose=2)
predict_train = model.predict([np.expand_dims(vox_train_mat, axis=-1), np.expand_dims(image_train_mat[:, 0, :, :], axis=-1), np.expand_dims(image_train_mat[:, 1, :, :], axis=-1), cali_train_mat], batch_size=10, verbose=2)

save_file = open(os.path.join(save_dir, save_file_name), 'wb')

output_dict = {}


output_dict['test_predictions'] = predict_test
output_dict['test_actual'] = label_test_mat

output_dict['train_predictions'] = predict_train
output_dict['train_actual'] = label_train_mat

pickle.dump(output_dict, save_file)

save_file.close()
