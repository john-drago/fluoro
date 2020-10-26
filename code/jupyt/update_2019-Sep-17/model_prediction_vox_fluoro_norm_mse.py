'''
This module will attempt to predict model parameters by using a trained model.
'''

import tensorflow as tf
import os
import h5py
import numpy as np
import pickle

base_dir = os.path.expanduser('~/fluoro/data/compilation')

file_base_name = 'vox_fluoro_norm_mse'

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

# -----------------------------------------------------------------


hist_file = open(os.path.join(hist_path, hist_file_name), 'rb')
hist_data = pickle.load(hist_file)

hist_file.close()

random_test_values = np.random.choice(hist_data['test_indxs'], size=predict_numb, replace=False)

vox_file = h5py.File(os.path.expanduser('~/fluoro/data/compilation/voxels_mark_origin_comp.h5py'), 'r')
vox_init = vox_file['vox_dset']
vox_test_mat = vox_init[sorted(random_test_values)]
vox_file.close()


image_file = h5py.File(os.path.expanduser('~/fluoro/data/compilation/images_norm_std.h5py'), 'r')
image_grp_1 = image_file['image_1']
image_grp_2 = image_file['image_2']
image_init_1 = image_grp_1['min_max_dset_per_image']
image_init_2 = image_grp_2['min_max_dset_per_image']

image_test_1 = image_init_1[sorted(random_test_values)]
image_test_2 = image_init_2[sorted(random_test_values)]
image_file.close()


label_file = h5py.File(os.path.expanduser('~/fluoro/data/compilation/labels.h5py'), 'r')
label_init = label_file['labels_dset']
label_test_mat = label_init[sorted(random_test_values)]
label_file.close()


cali_file = h5py.File(os.path.expanduser('~/fluoro/data/compilation/calibration_norm_std.h5py'), 'r')
cali_init = cali_file['min_max_dset']
cali_test_mat = cali_init[sorted(random_test_values)]
cali_file.close()

# -----------------------------------------------------------------

model = tf.keras.models.load_model(os.path.join(hist_path, load_model_name))


predict_1 = model.predict([np.expand_dims(vox_test_mat, axis=-1), image_test_1, image_test_2, cali_test_mat], batch_size=10, verbose=2)

save_file = open(os.path.join(save_dir, save_file_name), 'wb')

output_dict = {}

output_dict['raw_ouput'] = predict_1
output_dict['predictions'] = inv_min_max(predict_1, hist_data['label_train_val_min'], hist_data['label_train_val_max'])
output_dict['actual'] = label_test_mat

pickle.dump(output_dict, save_file)

save_file.close()
