import numpy as np
import h5py
import tensorflow as tf
# import keras
import os
# import sys
import pickle
from sklearn.model_selection import train_test_split

# sys.path.append(os.path.abspath(os.path.expanduser('~/fluoro/code')))

# import datacomp.h5py_multidimensional_array as h5py_multidimensional_array
# from datacomp.h5py_multidimensional_array import variable_matrix_loader


# expr_name = sys.argv[0][:-3]
expr_no = '1'
save_dir = os.path.abspath(os.path.expanduser('~/fluoro/code/jupyt/vox_fluoro'))
# os.makedirs(save_dir, exist_ok=True)


def data_comp(first_indx=None, last_indx=None):

    vox_file = h5py.File(os.path.expanduser('~/fluoro/data/compilation/voxels_pad.h5py'), 'r')
    vox_init = vox_file['vox_dset']
    vox_mat = vox_init[first_indx:last_indx]
    vox_file.close()

    image_file = h5py.File(os.path.expanduser('~/fluoro/data/compilation/images.h5py'), 'r')
    image_init = image_file['image_dset']
    image_mat = image_init[first_indx:last_indx]
    image_file.close()

    label_file = h5py.File(os.path.expanduser('~/fluoro/data/compilation/labels.h5py'), 'r')
    label_init = label_file['labels_dset']
    label_mat = label_init[first_indx:last_indx]
    label_file.close()

    cali_file = h5py.File(os.path.expanduser('~/fluoro/data/compilation/calibration.h5py'), 'r')
    cali_init = cali_file['cali_len3_rot']
    cali_mat = cali_init[first_indx:last_indx]
    cali_file.close()

    vox_train_cum, vox_test, image_train_cum, image_test, cali_train_cum, cali_test, label_train_cum, label_test = train_test_split(vox_mat, image_mat, cali_mat, label_mat, shuffle=True, test_size=0.2, random_state=42)

    # print('Image mat size:', image_mat.shape)
    # print('Label mat size:', label_mat.shape)
    # print('Cali mat size:', cali_mat.shape)

    # print('Image cum size:', image_train_cum.shape)
    # print('Label cum size:', label_train_cum.shape)
    # print('Cali cum size:', cali_train_cum.shape)

    # print('Image test size:', image_test.shape)
    # print('Label test size:', label_test.shape)
    # print('Cali test size:', cali_test.shape)


    vox_train_sub, vox_val, image_train_sub, image_val, cali_train_sub, cali_val, label_train_sub, label_val = train_test_split(vox_train_cum, image_train_cum, cali_train_cum, label_train_cum, shuffle=True, test_size=0.2, random_state=42)

    print('Image sub size:', image_train_sub.shape)
    print('Label sub size:', label_train_sub.shape)
    print('Cali sub size:', cali_train_sub.shape)

    print('Image val size:', image_val.shape)
    print('Label val size:', label_val.shape)
    print('Cali val size:', cali_val.shape)

    print(vox_mat.shape, image_mat.shape, cali_mat.shape)

    return vox_train_sub, vox_val, image_train_sub, image_val, cali_train_sub, cali_val, label_train_sub, label_val
    # return image_train_cum, cali_train_cum, label_train_cum


# -----------------------------------------------------------------

channel_order = 'channels_last'
img_input_shape = (128, 128, 1)
vox_input_shape = (198, 162, 564, 1)
cali_input_shape = (6,)


# def root_mean_squared_error(y_true, y_pred):
#     return tf.keras.backend.sqrt(tf.keras.backend.mean(tf.keras.backend.square(y_pred - y_true)))

# def root_mean_squared_error(y_true, y_pred):
#     base_dir = os.path.expanduser('~/fluoro/data/compilation')
#     stats_file = h5py.File(os.path.join(base_dir, 'labels_stats.h5py'), 'r')
#     mean_dset = stats_file['mean']
#     std_dset = stats_file['std']
#     var_dset = stats_file['var']
#     mean_v = mean_dset[:]
#     std_v = std_dset[:]
#     var_v = var_dset[:]

#     stats_file.close()


#     return tf.keras.backend.sqrt(tf.keras.backend.mean(tf.keras.backend.square(y_pred - y_true) / var_v))

def cust_mean_squared_error(y_true, y_pred):
    base_dir = os.path.expanduser('~/fluoro/data/compilation')
    stats_file = h5py.File(os.path.join(base_dir, 'labels_stats.h5py'), 'r')
    mean_dset = stats_file['mean']
    std_dset = stats_file['std']
    var_dset = stats_file['var']
    mean_v = mean_dset[:]
    std_v = std_dset[:]
    var_v = var_dset[:]

    stats_file.close()


    return tf.keras.backend.mean(((y_pred - y_true) / var_v)*1000)


params = {
    # 3D CONV
    'v_conv_1_filters': 30,
    'v_conv_1_kernel': 10,
    'v_conv_1_strides': 2,
    'v_conv_1_pad': 'same',
    'v_spatial_drop_rate_1': 0.5,
    'v_pool_1_size': 3,
    'v_pool_1_pad': 'valid',
    'v_conv_2_filters': 40,
    'v_conv_2_kernel': 5,
    'v_conv_2_strides': 2,
    'v_conv_2_pad': 'same',
    'v_spatial_drop_rate_2': 0.5,
    'v_pool_2_size': 2,
    'v_pool_2_pad': 'same',
    'v_conv_3_filters': 80,
    'v_conv_3_kernel': 3,
    'v_conv_3_strides': 2,
    'v_conv_3_pad': 'same',
    'v_spatial_drop_rate_3': 0.2,
    'v_pool_3_size': 2,
    'v_pool_3_pad': 'same',
    'dense_1_v_units': 1000,
    'dense_2_v_units': 500,
    'dense_3_v_units': 250,

    # 2D CONV
    'conv_1_filters': 30,
    'conv_1_kernel': 5,
    'conv_1_strides': 2,
    'conv_1_pad': 'same',
    'spatial_drop_rate_1': 0.5,
    'pool_1_size': 2,
    'pool_1_pad': 'same',
    'conv_2_filters': 40,
    'conv_2_kernel': 3,
    'conv_2_strides': 2,
    'conv_2_pad': 'same',
    'spatial_drop_rate_2': 0.5,
    'pool_2_size': 2,
    'pool_2_pad': 'same',
    'conv_3_filters': 80,
    'conv_3_kernel': 2,
    'conv_3_strides': 1,
    'conv_3_pad': 'same',
    'dense_1_f_units': 60,
    'dense_2_f_units': 60,
    'dense_3_f_units': 60,

    # Calibration Dense Layers
    'dense_1_cali_units': 10,
    'dense_2_cali_units': 10,

    # Top Level Dense Units
    'dense_1_co_units': 80,
    'drop_1_comb_rate': 0.2,
    'dense_2_co_units': 20,
    'dense_3_co_units': 20,
    'dense_4_co_units': 20,

    # Main Output
    'main_output_units': 6,
    'main_output_act': 'linear',

    # General Housekeeping
    'regularizer_l1': 0.1,
    'regularizer_l2': 0.25,
    'activation_fn': 'elu',
    'kern_init': 'glorot_uniform',
    'model_opt': 'adam',
    'learning_rate': 0.001,
    'model_epochs': 30,
    'model_batchsize': 5,
    'model_loss': cust_mean_squared_error,
    'model_metric': cust_mean_squared_error

}

# -----------------------------------------------------------------

# vox_ph_shape = list(vox_input_shape)
# img_ph_shape = list(img_input_shape)
# cali_ph_shape = list(cali_input_shape)

# vox_ph_shape.insert(0, 2)
# img_ph_shape.insert(0, 2)
# cali_ph_shape.insert(0, 2)

# vox_ph = tf.placeholder('float32', shape=vox_ph_shape)
# fluoro_1_ph = tf.placeholder('float16', shape=img_ph_shape)
# fluoro_2_ph = tf.placeholder('float16', shape=img_ph_shape)
# cali_ph = tf.placeholder('float16', shape=cali_ph_shape)

# input_vox = tf.keras.Input(shape=vox_input_shape, name='input_vox', tensor=vox_ph)
# input_fluoro_1 = tf.keras.Input(shape=img_input_shape, name='input_fluoro_1', tensor=fluoro_1_ph)
# input_fluoro_2 = tf.keras.Input(shape=img_input_shape, name='input_fluoro_2', tensor=fluoro_2_ph)
# input_cali = tf.keras.Input(shape=cali_input_shape, name='input_cali', tensor=cali_ph)

# -----------------------------------------------------------------

# Input Layers
input_vox = tf.keras.Input(shape=vox_input_shape, name='input_vox', dtype='float32')
input_fluoro_1 = tf.keras.Input(shape=img_input_shape, name='input_fluoro_1', dtype='float32')
input_fluoro_2 = tf.keras.Input(shape=img_input_shape, name='input_fluoro_2', dtype='float32')
input_cali = tf.keras.Input(shape=cali_input_shape, name='input_cali', dtype='float32')

# -----------------------------------------------------------------

# First run of 3D Conv Layers
# v_bn_1 = tf.keras.layers.BatchNormalization(input_shape=vox_input_shape)(input_vox)
v_conv_1 = tf.keras.layers.Conv3D(filters=params['v_conv_1_filters'], kernel_size=params['v_conv_1_kernel'], strides=params['v_conv_1_strides'], padding=params['v_conv_1_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'])(input_vox)
v_spat_1 = tf.keras.layers.SpatialDropout3D(rate=params['v_spatial_drop_rate_1'])(v_conv_1)
v_pool_1 = tf.keras.layers.MaxPooling3D(pool_size=params['v_pool_1_size'], padding=params['v_pool_1_pad'], data_format=channel_order)(v_spat_1)

# Second run of 3D Conv Layers
v_bn_2 = tf.keras.layers.BatchNormalization()(v_pool_1)
v_conv_2 = tf.keras.layers.Conv3D(filters=params['v_conv_2_filters'], kernel_size=params['v_conv_2_kernel'], strides=params['v_conv_2_strides'], padding=params['v_conv_2_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'])(v_bn_2)
v_spat_2 = tf.keras.layers.SpatialDropout3D(rate=params['v_spatial_drop_rate_2'])(v_conv_2)
v_pool_2 = tf.keras.layers.MaxPooling3D(pool_size=params['v_pool_2_size'], padding=params['v_pool_2_pad'], data_format=channel_order)(v_spat_2)

# Third run of 3D Conv Layers
v_conv_3 = tf.keras.layers.Conv3D(filters=params['v_conv_3_filters'], kernel_size=params['v_conv_3_kernel'], strides=params['v_conv_3_strides'], padding=params['v_conv_3_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'])(v_pool_2)
v_spat_3 = tf.keras.layers.SpatialDropout3D(rate=params['v_spatial_drop_rate_3'])(v_conv_3)
v_pool_3 = tf.keras.layers.MaxPooling3D(pool_size=params['v_pool_3_size'], padding=params['v_pool_3_pad'], data_format=channel_order)(v_spat_3)

v_flatten_1 = tf.keras.layers.Flatten()(v_pool_3)

# Dense Layers After Flattended 3D Conv
dense_1_v = tf.keras.layers.Dense(units=params['dense_1_v_units'], activation=params['activation_fn'], kernel_initializer=params['kern_init'])(v_flatten_1)
dense_2_v = tf.keras.layers.Dense(units=params['dense_2_v_units'], activation=params['activation_fn'], kernel_initializer=params['kern_init'])(dense_1_v)
dense_3_v = tf.keras.layers.Dense(units=params['dense_3_v_units'], activation=params['activation_fn'], kernel_initializer=params['kern_init'])(dense_2_v)

# -----------------------------------------------------------------

# First run of 2D Conv Layers for Image 1
bn_1_1 = tf.keras.layers.BatchNormalization(input_shape=img_input_shape)(input_fluoro_1)
conv_1_1 = tf.keras.layers.Conv2D(filters=params['conv_1_filters'], kernel_size=params['conv_1_kernel'], strides=params['conv_1_strides'], padding=params['conv_1_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'])(bn_1_1)
spat_1_1 = tf.keras.layers.SpatialDropout2D(rate=params['spatial_drop_rate_1'])(conv_1_1)
pool_1_1 = tf.keras.layers.MaxPooling2D(pool_size=params['pool_1_size'], padding=params['pool_1_pad'], data_format=channel_order)(spat_1_1)

# Second run of 2D Conv Layers for Image 1
conv_2_1 = tf.keras.layers.Conv2D(filters=params['conv_2_filters'], kernel_size=params['conv_2_kernel'], strides=params['conv_2_strides'], padding=params['conv_2_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'])(pool_1_1)
spat_2_1 = tf.keras.layers.SpatialDropout2D(rate=params['spatial_drop_rate_2'])(conv_2_1)
pool_2_1 = tf.keras.layers.MaxPooling2D(pool_size=params['pool_2_size'], padding=params['pool_2_pad'], data_format=channel_order)(spat_2_1)

# Third run of 2D Conv Layers for Image 1
conv_3_1 = tf.keras.layers.Conv2D(filters=params['conv_2_filters'], kernel_size=params['conv_2_kernel'], strides=params['conv_2_strides'], padding=params['conv_2_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'])(pool_2_1)
pool_3_1 = tf.keras.layers.MaxPooling2D(pool_size=params['pool_2_size'], padding=params['pool_2_pad'], data_format=channel_order)(conv_3_1)

flatten_1_1 = tf.keras.layers.Flatten()(pool_3_1)

# Dense Layers After Flattended 2D Conv
dense_1_f_1 = tf.keras.layers.Dense(units=params['dense_1_f_units'], activation=params['activation_fn'], kernel_initializer=params['kern_init'])(flatten_1_1)
dense_2_f_1 = tf.keras.layers.Dense(units=params['dense_2_f_units'], activation=params['activation_fn'], kernel_initializer=params['kern_init'])(dense_1_f_1)
dense_3_f_1 = tf.keras.layers.Dense(units=params['dense_3_f_units'], activation=params['activation_fn'], kernel_initializer=params['kern_init'])(dense_2_f_1)

# -----------------------------------------------------------------

# First run of 2D Conv Layers for Image 2
bn_1_2 = tf.keras.layers.BatchNormalization(input_shape=img_input_shape)(input_fluoro_2)
conv_1_2 = tf.keras.layers.Conv2D(filters=params['conv_1_filters'], kernel_size=params['conv_1_kernel'], strides=params['conv_1_strides'], padding=params['conv_1_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'])(bn_1_2)
spat_1_2 = tf.keras.layers.SpatialDropout2D(rate=params['spatial_drop_rate_1'])(conv_1_2)
pool_1_2 = tf.keras.layers.MaxPooling2D(pool_size=params['pool_1_size'], padding=params['pool_1_pad'], data_format=channel_order)(spat_1_2)

# Second run of 2D Conv Layers for Image 1
conv_2_2 = tf.keras.layers.Conv2D(filters=params['conv_2_filters'], kernel_size=params['conv_2_kernel'], strides=params['conv_2_strides'], padding=params['conv_2_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'])(pool_1_2)
spat_2_2 = tf.keras.layers.SpatialDropout2D(rate=params['spatial_drop_rate_2'])(conv_2_2)
pool_2_2 = tf.keras.layers.MaxPooling2D(pool_size=params['pool_2_size'], padding=params['pool_2_pad'], data_format=channel_order)(spat_2_2)

# Third run of 2D Conv Layers for Image 1
conv_3_2 = tf.keras.layers.Conv2D(filters=params['conv_2_filters'], kernel_size=params['conv_2_kernel'], strides=params['conv_2_strides'], padding=params['conv_2_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'])(pool_2_2)
pool_3_2 = tf.keras.layers.MaxPooling2D(pool_size=params['pool_2_size'], padding=params['pool_2_pad'], data_format=channel_order)(conv_3_2)

flatten_1_2 = tf.keras.layers.Flatten()(pool_3_2)

# Dense Layers After Flattended 2D Conv
dense_1_f_2 = tf.keras.layers.Dense(units=params['dense_1_f_units'], activation=params['activation_fn'], kernel_initializer=params['kern_init'])(flatten_1_2)
dense_2_f_2 = tf.keras.layers.Dense(units=params['dense_2_f_units'], activation=params['activation_fn'], kernel_initializer=params['kern_init'])(dense_1_f_2)
dense_3_f_2 = tf.keras.layers.Dense(units=params['dense_3_f_units'], activation=params['activation_fn'], kernel_initializer=params['kern_init'])(dense_2_f_2)

# -----------------------------------------------------------------

# Dense Layers Over Calibration Data
dense_1_cali = tf.keras.layers.Dense(units=params['dense_1_cali_units'], activation=params['activation_fn'], kernel_initializer=params['kern_init'])(input_cali)
dense_2_cali = tf.keras.layers.Dense(units=params['dense_2_cali_units'], activation=params['activation_fn'], kernel_initializer=params['kern_init'])(dense_1_cali)

# -----------------------------------------------------------------

# Combine Vox Data, Fluoro Data, and Cali Data
dense_0_comb = tf.keras.layers.concatenate([dense_3_v, dense_3_f_1, dense_3_f_2, dense_2_cali])

# -----------------------------------------------------------------

# Dense Layers at Top of Model
dense_1_comb = tf.keras.layers.Dense(units=params['dense_1_co_units'], activation=params['activation_fn'], kernel_initializer=params['kern_init'])(dense_0_comb)
dense_drop_1 = tf.keras.layers.Dropout(rate=params['drop_1_comb_rate'])(dense_1_comb)
dense_2_comb = tf.keras.layers.Dense(units=params['dense_2_co_units'], activation=params['activation_fn'], kernel_initializer=params['kern_init'])(dense_drop_1)
dense_3_comb = tf.keras.layers.Dense(units=params['dense_3_co_units'], activation=params['activation_fn'], kernel_initializer=params['kern_init'])(dense_2_comb)
dense_4_comb = tf.keras.layers.Dense(units=params['dense_4_co_units'], activation=params['activation_fn'], kernel_initializer=params['kern_init'])(dense_3_comb)

# -----------------------------------------------------------------

# Main Output
main_output = tf.keras.layers.Dense(units=params['main_output_units'], activation=params['main_output_act'], kernel_initializer=params['kern_init'], name='main_output')(dense_4_comb)

# -----------------------------------------------------------------

# Model Housekeeping
model = tf.keras.Model(inputs=[input_vox, input_fluoro_1, input_fluoro_2, input_cali], outputs=main_output)

model.compile(optimizer=params['model_opt'], loss=params['model_loss'], metrics=[params['model_metric']])
# tf.keras.utils.plot_model(model, os.path.abspath(os.path.join(save_dir, expr_name + '_' + expr_no + '.png')), show_shapes=True)


label_file = h5py.File(os.path.expanduser('~/fluoro/data/compilation/labels.h5py'), 'r')
label_init = label_file['labels_dset']
label_mat = label_init[:]

total_num_of_samples = label_mat.shape[0]
label_file.close()

print('num of train samples:', total_num_of_samples)



subset_size = 2000

hist_file = open('vox_fluoro_hist_objects', 'wb')
hist_dict = {}

for subset_iter in range(int(np.ceil(total_num_of_samples / subset_size))):

    print('First_index:\t', subset_iter * subset_size)
    print('Last_index:\t', subset_iter * subset_size + subset_size)

    vox_train_sub, vox_val, image_train_sub, image_val, cali_train_sub, cali_val, label_train_sub, label_val = data_comp(first_indx=subset_iter * subset_size, last_indx=subset_iter * subset_size + subset_size)

    print('\n\n\n\nGot past the load:\n\n\n\n')

    result = model.fit(x={'input_vox': np.expand_dims(vox_train_sub[:], axis=-1), 'input_fluoro_1': np.expand_dims(image_train_sub[:, 0, :, :], axis=-1), 'input_fluoro_2': np.expand_dims(image_train_sub[:, 1, :, :], axis=-1), 'input_cali': cali_train_sub[:]}, y=label_train_sub[:], validation_data=([np.expand_dims(vox_val, axis=-1), np.expand_dims(image_val[:, 0, :, :], axis=-1), np.expand_dims(image_val[:, 1, :, :], axis=-1), cali_val], label_val), epochs=params['model_epochs'], batch_size=params['model_batchsize'], shuffle=True, verbose=2)
    hist_dict['result' + '_' + str(subset_iter)] = result

pickle.dump(hist_dict, hist_file)

hist_file.close()




# result = model.fit(x={'input_vox': np.expand_dims(vox_train_sub, axis=-1), 'input_fluoro_1': np.expand_dims(image_train_sub[:, 0, :, :], axis=-1), 'input_fluoro_2': np.expand_dims(image_train_sub[:, 1, :, :], axis=-1), 'input_cali': cali_train_sub}, y=label_train_sub, validation_data=([np.expand_dims(vox_val, axis=-1), np.expand_dims(image_val[:, 0, :, :], axis=-1), np.expand_dims(image_val[:, 1, :, :], axis=-1), cali_val], label_val), epochs=params['model_epochs'], batch_size=params['model_batchsize'], shuffle=True, verbose=2)

model.save(os.path.join(os.getcwd(), 'vox_fluoro_model_save' + '_' + expr_no + '.h5'))



