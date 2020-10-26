'''
This module will attempt to predict model parameters by using a trained model.
'''

import tensorflow as tf
import os
import h5py
import numpy as np
import pickle

base_dir = os.path.expanduser('~/fluoro/data/compilation')

file_base_name = 'vox_fluoro_min_max_1'

hist_path = os.path.join(os.path.expanduser('~/fluoro/code/jupyt/vox_fluoro'), file_base_name)
hist_file_name = file_base_name + '_hist_objects_1.pkl'
load_model_name = file_base_name + '_1.h5'

save_dir = os.path.abspath(os.path.expanduser('~/fluoro/code/jupyt/update_2019-Sep-17/predictions'))
os.makedirs(save_dir, exist_ok=True)

save_file_name = 'model_prediction_' + file_base_name + '.pkl'

predict_numb = 100

# -----------------------------------------------------------------


def inv_min_max(data_set, data_min, data_max, axis=0):

    data_0_1 = (data_set - np.min(data_set, axis=axis)) / (np.max(data_set, axis=axis) - np.min(data_set, axis=axis))
    inv_data = data_0_1 * (data_max - data_min) + data_min

    inv_data = np.where(inv_data < data_min, data_min, inv_data)
    inv_data = np.where(inv_data > data_max, data_max, inv_data)
    return inv_data


def min_max_norm(data_set, feature_range=(-1, 1), axis=0, data_min=None, data_max=None):

    if data_min is None:
        data_min = np.min(data_set, axis=axis)
    else:
        data_set = np.where(data_set < data_min, data_min, data_set)

    if data_max is None:
        data_max = np.max(data_set, axis=axis)
    else:
        data_set = np.where(data_set > data_max, data_max, data_set)


    data_in_std_range = (data_set - data_min) / (data_max - data_min)

    data_scaled = data_in_std_range * (feature_range[1] - feature_range[0]) + feature_range[0]

    return data_scaled



# -----------------------------------------------------------------



hist_file = open(os.path.join(hist_path, hist_file_name), 'rb')
hist_data = pickle.load(hist_file)

hist_file.close()

random_test_values = np.random.choice(hist_data['test_indxs'], size=predict_numb, replace=False)
random_train_values = np.random.choice(hist_data['train_indxs'], size=predict_numb, replace=False)

vox_file = h5py.File(os.path.expanduser('~/fluoro/data/compilation/voxels_mark_origin_comp.h5py'), 'r')
vox_init = vox_file['vox_dset']
vox_test_mat = vox_init[sorted(random_test_values)]
vox_train_mat = vox_init[sorted(random_train_values)]
vox_file.close()


image_file = h5py.File(os.path.expanduser('~/fluoro/data/compilation/images_norm_std.h5py'), 'r')
image_grp_1 = image_file['image_1']
image_grp_2 = image_file['image_2']
image_init_1 = image_grp_1['min_max_dset_per_image']
image_init_2 = image_grp_2['min_max_dset_per_image']

image_test_1 = image_init_1[sorted(random_test_values)]
image_test_2 = image_init_2[sorted(random_test_values)]

image_train_1 = image_init_1[sorted(random_train_values)]
image_train_2 = image_init_2[sorted(random_train_values)]

image_file.close()


label_file = h5py.File(os.path.expanduser('~/fluoro/data/compilation/labels.h5py'), 'r')
label_init = label_file['labels_dset']
label_test_mat = label_init[sorted(random_test_values)]
label_train_mat = label_init[sorted(random_train_values)]
label_file.close()


cali_file = h5py.File(os.path.expanduser('~/fluoro/data/compilation/calibration.h5py'), 'r')
cali_init = cali_file['cali_len3_rot']
cali_test_mat = cali_init[sorted(random_test_values)]
cali_test_min_max = min_max_norm(cali_test_mat, data_min=hist_data['cali_train_min'], data_max=hist_data['cali_train_max'])
cali_train_mat = cali_init[sorted(random_train_values)]
cali_train_min_max = min_max_norm(cali_train_mat, data_min=hist_data['cali_train_min'], data_max=hist_data['cali_train_max'])
cali_file.close()

# -----------------------------------------------------------------

model = tf.keras.models.load_model(os.path.join(hist_path, load_model_name))

predict_test = model.predict([np.expand_dims(vox_test_mat, axis=-1), image_test_1, image_test_2, cali_test_mat], batch_size=6, verbose=2)
predict_train = model.predict([np.expand_dims(vox_train_mat, axis=-1), image_train_1, image_train_2, cali_train_mat], batch_size=6, verbose=2)

save_file = open(os.path.join(save_dir, save_file_name), 'wb')

output_dict = {}

output_dict['test_raw_ouput'] = predict_test
output_dict['test_predictions'] = inv_min_max(predict_test, hist_data['label_train_min'], hist_data['label_train_max'])
output_dict['test_actual'] = label_test_mat


output_dict['train_raw_ouput'] = predict_train
output_dict['train_predictions'] = inv_min_max(predict_train, hist_data['label_train_min'], hist_data['label_train_max'])
output_dict['train_actual'] = label_train_mat

pickle.dump(output_dict, save_file)

save_file.close()
