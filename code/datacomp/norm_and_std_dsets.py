'''
This module will generate the necessary data points for all of the datasets (for training fluoroscopic neural net) in order to perform normalization and standardization.
'''
import numpy as np
import h5py
import os
import time

# -----------------------------------------------------------------


load_dir = '/Volumes/Seagate/fluoro'
save_dir = '/Volumes/Seagate/fluoro'

# -----------------------------------------------------------------


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


def min_max_norm_per_image(data_set, feature_range=(-1, 1), axis=(1, 2)):

    data_min = np.min(data_set, axis=axis)
    data_max = np.max(data_set, axis=axis)

    while (len(data_min.shape) < len(data_set.shape)) and (len(data_max.shape) < len(data_set.shape)):
        # print('Min shape:', data_min.shape)
        # print('Max shape:', data_max.shape)
        data_min = np.expand_dims(data_min, axis=1)
        data_max = np.expand_dims(data_max, axis=1)

    data_in_std_range = (data_set - data_min) / (data_max - data_min)

    data_scaled = data_in_std_range * (feature_range[1] - feature_range[0]) + feature_range[0]

    return data_scaled


def inv_min_max(data_set, data_min, data_max, axis=0):

    data_0_1 = (data_set - np.min(data_set, axis=axis)) / (np.max(data_set, axis=axis) - np.min(data_set, axis=axis))
    inv_data = data_0_1 * (data_max - data_min) + data_min

    inv_data = np.where(inv_data < data_min, data_min, inv_data)
    inv_data = np.where(inv_data > data_max, data_max, inv_data)
    return inv_data


def feature_scaler(data_set, mean=0, std=1, axis=0):
    data_set_0_1 = (data_set - np.mean(data_set, axis=axis)) / (np.std(data_set, axis=axis) / std) + mean
    return data_set_0_1


def feature_scaler_per_image(data_set, mean=0, std=1, axis=(1, 2)):

    data_mean = np.mean(data_set, axis=axis)
    data_std = np.std(data_set, axis=axis)

    while (len(data_mean.shape) < len(data_set.shape)) and (len(data_std.shape) < len(data_set.shape)):
        # print('Min shape:', data_mean.shape)
        # print('Max shape:', data_std.shape)
        data_mean = np.expand_dims(data_mean, axis=1)
        data_std = np.expand_dims(data_std, axis=1)

    data_set_0_1 = (data_set - data_mean) / (data_std / std) + mean
    return data_set_0_1


def inv_feature_scaler(data_set, data_mean, data_std, axis=0):
    inv_data = (data_set - np.mean(data_set, axis=axis)) * (data_std / np.std(data_set, axis=axis)) + data_mean
    return inv_data

# -----------------------------------------------------------------


# labels_file = h5py.File(os.path.expanduser('~/fluoro/data/compilation/labels.h5py'), 'r')
# label_norm_std_file = h5py.File(os.path.join(save_dir, 'labels_norm_std.h5py'), 'w')


# labels_mat = labels_file['labels_dset']

# labels_dset = labels_mat[:]

# labels_std = np.std(labels_dset, axis=0)
# labels_mean = np.mean(labels_dset, axis=0)
# labels_var = np.var(labels_dset, axis=0)
# labels_min = np.min(labels_dset, axis=0)
# labels_max = np.max(labels_dset, axis=0)
# labels_min_max_norm = min_max_norm(labels_dset)
# labels_std_scale = feature_scaler(labels_dset)


# std_dset = label_norm_std_file.create_dataset('std', data=labels_std)
# mean_dset = label_norm_std_file.create_dataset('mean', data=labels_mean)
# var_dset = label_norm_std_file.create_dataset('var', data=labels_var)
# min_dset = label_norm_std_file.create_dataset('min', data=labels_min)
# max_dset = label_norm_std_file.create_dataset('max', data=labels_max)
# min_max_norm_dset = label_norm_std_file.create_dataset('min_max_dset', data=labels_min_max_norm)
# std_scale_dset = label_norm_std_file.create_dataset('std_scale_dset', data=labels_std_scale)

# label_norm_std_file.close()
# labels_file.close()


# -----------------------------------------------------------------


# cali_file = h5py.File(os.path.expanduser('~/fluoro/data/compilation/calibration.h5py'), 'r')
# cali_norm_std_file = h5py.File(os.path.join(save_dir, 'calibration_norm_std.h5py'), 'w')


# cali_mat = cali_file['cali_len3_rot']

# cali_dset = cali_mat[:]

# cali_std = np.std(cali_dset, axis=0)
# cali_mean = np.mean(cali_dset, axis=0)
# cali_var = np.var(cali_dset, axis=0)
# cali_min = np.min(cali_dset, axis=0)
# cali_max = np.max(cali_dset, axis=0)
# cali_min_max_norm = min_max_norm(cali_dset)
# cali_std_scale = feature_scaler(cali_dset)


# std_dset = cali_norm_std_file.create_dataset('std', data=cali_std)
# mean_dset = cali_norm_std_file.create_dataset('mean', data=cali_mean)
# var_dset = cali_norm_std_file.create_dataset('var', data=cali_var)
# min_dset = cali_norm_std_file.create_dataset('min', data=cali_min)
# max_dset = cali_norm_std_file.create_dataset('max', data=cali_max)
# min_max_norm_dset = cali_norm_std_file.create_dataset('min_max_dset', data=cali_min_max_norm)
# std_scale_dset = cali_norm_std_file.create_dataset('std_scale_dset', data=cali_std_scale)

# cali_norm_std_file.close()
# cali_file.close()


# -----------------------------------------------------------------


# image_file = h5py.File(os.path.expanduser('~/fluoro/data/compilation/images.h5py'), 'r')
# image_norm_std_file = h5py.File(os.path.join(save_dir, 'images_norm_std.h5py'), 'w')


# image_mat = image_file['image_dset']

# image_dset_1 = np.expand_dims(image_mat[:, 0, :, :], axis=-1)

# image_std_1 = np.std(image_dset_1, axis=(1, 2))
# image_mean_1 = np.mean(image_dset_1, axis=(1, 2))
# image_var_1 = np.var(image_dset_1, axis=(1, 2))
# image_min_1 = np.min(image_dset_1, axis=(1, 2))
# image_max_1 = np.max(image_dset_1, axis=(1, 2))
# image_min_max_norm_1 = min_max_norm_per_image(image_dset_1)
# image_std_scale_1 = feature_scaler_per_image(image_dset_1)

# fluoro_1 = image_norm_std_file.create_group('image_1')


# std_dset = fluoro_1.create_dataset('std_per_image', data=image_std_1)
# mean_dset = fluoro_1.create_dataset('mean_per_image', data=image_mean_1)
# var_dset = fluoro_1.create_dataset('var_per_image', data=image_var_1)
# min_dset = fluoro_1.create_dataset('min_per_image', data=image_min_1)
# max_dset = fluoro_1.create_dataset('max_per_image', data=image_max_1)
# min_max_norm_dset = fluoro_1.create_dataset('min_max_dset_per_image', data=image_min_max_norm_1)
# std_scale_dset = fluoro_1.create_dataset('std_scale_dset_per_image', data=image_std_scale_1)


# image_dset_2 = np.expand_dims(image_mat[:, 1, :, :], axis=-1)

# image_std_2 = np.std(image_dset_2, axis=(1, 2))
# image_mean_2 = np.mean(image_dset_2, axis=(1, 2))
# image_var_2 = np.var(image_dset_2, axis=(1, 2))
# image_min_2 = np.min(image_dset_2, axis=(1, 2))
# image_max_2 = np.max(image_dset_2, axis=(1, 2))
# image_min_max_norm_2 = min_max_norm_per_image(image_dset_2)
# image_std_scale_2 = feature_scaler_per_image(image_dset_2)

# fluoro_2 = image_norm_std_file.create_group('image_2')


# std_dset = fluoro_2.create_dataset('std_per_image', data=image_std_2)
# mean_dset = fluoro_2.create_dataset('mean_per_image', data=image_mean_2)
# var_dset = fluoro_2.create_dataset('var_per_image', data=image_var_2)
# min_dset = fluoro_2.create_dataset('min_per_image', data=image_min_2)
# max_dset = fluoro_2.create_dataset('max_per_image', data=image_max_2)
# min_max_norm_dset = fluoro_2.create_dataset('min_max_dset_per_image', data=image_min_max_norm_2)
# std_scale_dset = fluoro_2.create_dataset('std_scale_dset_per_image', data=image_std_scale_2)


# image_norm_std_file.close()
# image_file.close()

# -----------------------------------------------------------------

# vox_file = h5py.File(os.path.expanduser('~/fluoro/data/compilation/voxels_mark_origin_comp.h5py'), 'r')
# vox_norm_std_file = h5py.File(os.path.join(save_dir, 'voxels_norm_std.h5py'), 'w')

# vox_mat = vox_file['vox_dset']

# vox_dset = np.expand_dims(vox_mat, axis=-1)

# vox_std = np.std(vox_dset, axis=(1, 2, 3))
# vox_mean = np.mean(vox_dset, axis=(1, 2, 3))
# vox_var = np.var(vox_dset, axis=(1, 2, 3))
# vox_min = np.min(vox_dset, axis=(1, 2, 3))
# vox_max = np.max(vox_dset, axis=(1, 2, 3))
# vox_min_max_norm = min_max_norm_per_image(vox_dset, axis=(1, 2, 3))
# vox_std_scale = feature_scaler_per_image(vox_dset, axis=(1, 2, 3))


# std_dset = vox_norm_std_file.create_dataset('std', data=vox_std)
# mean_dset = vox_norm_std_file.create_dataset('mean', data=vox_mean)
# var_dset = vox_norm_std_file.create_dataset('var', data=vox_var)
# min_dset = vox_norm_std_file.create_dataset('min', data=vox_min)
# max_dset = vox_norm_std_file.create_dataset('max', data=vox_max)
# min_max_norm_dset = vox_norm_std_file.create_dataset('min_max_dset', data=vox_min_max_norm, compression='lzf')
# std_scale_dset = vox_norm_std_file.create_dataset('std_scale_dset', data=vox_std_scale, compression='lzf')

# vox_norm_std_file.close()
# vox_file.close()
