import numpy as np
import h5py
import tensorflow as tf
import os
import sys
import pickle
import datetime


# 2019-09-30
# In this file we are going to complete unit testing to see where the current model goes wrong.

# We are not initially going to use batch normalization or dropout.


expr_name = sys.argv[0][:-3]
save_dir = os.path.abspath(os.getcwd())
os.makedirs(save_dir, exist_ok=True)

save_image = True

# -----------------------------------------------------------------
# Initialize TensorBoard / Keras Callbacks information

root_logdir = os.path.join(save_dir, 'tf_logs')
root_pydir = os.path.join(save_dir, 'py_hist')

run_id = datetime.datetime.now().strftime("%Y_%m_%d__%H%M%S")
log_dir = os.path.join(root_logdir, 'run_' + run_id)
py_dir = os.path.join(root_pydir, 'run_' + run_id)

os.makedirs(py_dir, exist_ok=True)

tf.debugging.set_log_device_placement(True)


# -----------------------------------------------------------------
# This part of the file will create a replica of the python file so that we can have a running history of python scripts that have been run.

with open(expr_name + '.py') as f:
    with open(os.path.join(py_dir, 'run_' + run_id + '.py'), 'w') as f1:
        for line in f:
            f1.write(line)


# -----------------------------------------------------------------


def split_train_test(shape, num_of_samples=None, ratio=0.2):

    np.random.seed(np.random.choice(2**16))

    if num_of_samples is None:
        shuffled_indices = np.random.choice(shape, size=shape, replace=False)
    else:
        shuffled_indices = np.random.choice(shape, size=num_of_samples, replace=False)


    test_set_size = int(len(shuffled_indices) * 0.2)
    test_indx = shuffled_indices[:test_set_size]
    train_indx = shuffled_indices[test_set_size:]

    return test_indx, train_indx



# -----------------------------------------------------------------
# This is the file, which will help us normalize our input data to common ranges.



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


# -----------------------------------------------------------------



params = {

    # 3D CONV Layers
    'v_conv_0_filters': 30,
    'v_conv_0_kernel': 7,
    'v_conv_0_strides': 2,
    'v_conv_0_pad': 'same',

    'v_conv_1_filters': 30,
    'v_conv_1_kernel': 5,
    'v_conv_1_strides_0': 2,
    'v_conv_1_strides_1': 2,
    'v_conv_1_strides_2': 3,
    'v_conv_1_pad': 'same',

    'v_pool_0_size': 2,
    'v_pool_0_pad': 'same',

    'v_conv_2_filters': 30,
    'v_conv_2_kernel': 3,
    'v_conv_2_strides': 2,
    'v_conv_2_pad': 'same',

    'v_conv_3_filters': 30,
    'v_conv_3_kernel': 3,
    'v_conv_3_strides': 1,
    'v_conv_3_pad': 'same',

    'v_conv_4_filters': 40,
    'v_conv_4_kernel': 3,
    'v_conv_4_strides': 2,
    'v_conv_4_pad': 'same',

    'v_conv_5_filters': 40,
    'v_conv_5_kernel': 3,
    'v_conv_5_strides': 1,
    'v_conv_5_pad': 'same',

    'v_conv_6_filters': 40,
    'v_conv_6_kernel': 3,
    'v_conv_6_strides': 1,
    'v_conv_6_pad': 'same',

    'v_conv_7_filters': 50,
    'v_conv_7_kernel': 3,
    'v_conv_7_strides': 1,
    'v_conv_7_pad': 'same',

    'v_conv_8_filters': 50,
    'v_conv_8_kernel': 2,
    'v_conv_8_strides': 2,
    'v_conv_8_pad': 'same',

    'v_conv_9_filters': 50,
    'v_conv_9_kernel': 2,
    'v_conv_9_strides': 2,
    'v_conv_9_pad': 'same',

    'v_conv_10_filters': 60,
    'v_conv_10_kernel': 2,
    'v_conv_10_strides': 1,
    'v_conv_10_pad': 'valid',

    'dense_0_vox_units': 100,
    'dense_1_vox_units': 80,
    'dense_2_vox_units': 80,


    # 2D CONV Layers
    'conv_0_filters': 30,
    'conv_0_kernel': 5,
    'conv_0_strides': 2,
    'conv_0_pad': 'same',

    'conv_1_filters': 40,
    'conv_1_kernel': 3,
    'conv_1_strides': 1,
    'conv_1_pad': 'same',

    'pool_0_size': 2,
    'pool_0_pad': 'same',

    'conv_2_filters': 40,
    'conv_2_kernel': 3,
    'conv_2_strides': 1,
    'conv_2_pad': 'same',

    'conv_3_filters': 40,
    'conv_3_kernel': 3,
    'conv_3_strides': 2,
    'conv_3_pad': 'same',

    'conv_4_filters': 40,
    'conv_4_kernel': 3,
    'conv_4_strides': 2,
    'conv_4_pad': 'same',

    'conv_5_filters': 40,
    'conv_5_kernel': 2,
    'conv_5_strides': 1,
    'conv_5_pad': 'same',

    'conv_6_filters': 40,
    'conv_6_kernel': 2,
    'conv_6_strides': 2,
    'conv_6_pad': 'valid',

    'conv_7_filters': 40,
    'conv_7_kernel': 2,
    'conv_7_strides': 2,
    'conv_7_pad': 'valid',

    'conv_8_filters': 40,
    'conv_8_kernel': 2,
    'conv_8_strides': 1,
    'conv_8_pad': 'valid',

    'dense_0_flu_units': 80,
    'dense_1_flu_units': 80,
    'dense_2_flu_units': 80,

    # Cali Dense Units
    'dense_0_cali_units': 100,
    'dense_1_cali_units': 100,
    'dense_2_cali_units': 100,
    'dense_3_cali_units': 100,
    'dense_4_cali_units': 50,


    # Top Level Dense Units
    'dense_1_co_units': 300,
    'dense_2_co_units': 250,
    'dense_3_co_units': 200,
    'dense_4_co_units': 150,
    'dense_5_co_units': 100,
    'dense_6_co_units': 100,
    'dense_7_co_units': 100,
    'dense_8_co_units': 6,

    'drop_1_comb_rate': 0.3,
    'drop_2_comb_rate': 0.3,

    # Main Output
    'main_output_units': 6,
    'main_output_act': 'linear',

    'act_reg': None,
    'activation_fn': 'elu',
    'kern_init': 'he_uniform',
    'model_opt': tf.keras.optimizers.Adam,
    'learning_rate': 0.001,

    'model_epochs': 100,
    'model_batchsize': 5,
    'model_loss': 'mse',
    'model_metric': 'mse'
}

vox_input_shape = (197, 162, 564, 1)
cali_input_shape = (6,)
img_input_shape = (128, 128, 1)
channel_order = 'channels_last'


# -----------------------------------------------------------------
# Input Layers

input_vox = tf.keras.layers.Input(shape=vox_input_shape, name='input_vox', dtype='float32')

input_cali = tf.keras.layers.Input(shape=cali_input_shape, name='input_cali', dtype='float32')

input_fluoro_1 = tf.keras.layers.Input(shape=img_input_shape, name='input_fluoro_1', dtype='float32')
input_fluoro_2 = tf.keras.layers.Input(shape=img_input_shape, name='input_fluoro_2', dtype='float32')

# input_cali_1 = tf.keras.backend.print_tensor(input_cali)


# -----------------------------------------------------------------
# Voxel Analysis

v_conv_0 = tf.keras.layers.Conv3D(filters=params['v_conv_0_filters'], kernel_size=params['v_conv_0_kernel'], strides=params['v_conv_0_strides'], padding=params['v_conv_0_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['act_reg'])(input_vox)
bn_0 = tf.keras.layers.BatchNormalization()(v_conv_0)

v_conv_1 = tf.keras.layers.Conv3D(filters=params['v_conv_1_filters'], kernel_size=params['v_conv_1_kernel'], strides=(params['v_conv_1_strides_0'], params['v_conv_1_strides_1'], params['v_conv_1_strides_2']), padding=params['v_conv_1_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['act_reg'])(bn_0)
bn_1 = tf.keras.layers.BatchNormalization()(v_conv_1)

pool_0 = tf.keras.layers.MaxPooling3D(pool_size=params['v_pool_0_size'], padding=params['v_pool_0_pad'])(bn_1)

v_conv_2 = tf.keras.layers.Conv3D(filters=params['v_conv_2_filters'], kernel_size=params['v_conv_2_kernel'], strides=params['v_conv_2_strides'], padding=params['v_conv_2_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['act_reg'])(pool_0)
bn_2 = tf.keras.layers.BatchNormalization()(v_conv_2)

v_conv_3 = tf.keras.layers.Conv3D(filters=params['v_conv_3_filters'], kernel_size=params['v_conv_3_kernel'], strides=params['v_conv_3_strides'], padding=params['v_conv_3_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['act_reg'])(bn_2)
bn_3 = tf.keras.layers.BatchNormalization()(v_conv_3)

v_conv_4 = tf.keras.layers.Conv3D(filters=params['v_conv_4_filters'], kernel_size=params['v_conv_4_kernel'], strides=params['v_conv_4_strides'], padding=params['v_conv_4_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['act_reg'])(bn_3)
bn_4 = tf.keras.layers.BatchNormalization()(v_conv_4)

v_conv_5 = tf.keras.layers.Conv3D(filters=params['v_conv_5_filters'], kernel_size=params['v_conv_5_kernel'], strides=params['v_conv_5_strides'], padding=params['v_conv_5_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['act_reg'])(bn_4)
bn_5 = tf.keras.layers.BatchNormalization()(v_conv_5)

v_conv_6 = tf.keras.layers.Conv3D(filters=params['v_conv_6_filters'], kernel_size=params['v_conv_6_kernel'], strides=params['v_conv_6_strides'], padding=params['v_conv_6_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['act_reg'])(bn_5)
bn_6 = tf.keras.layers.BatchNormalization()(v_conv_6)

v_conv_7 = tf.keras.layers.Conv3D(filters=params['v_conv_7_filters'], kernel_size=params['v_conv_7_kernel'], strides=params['v_conv_7_strides'], padding=params['v_conv_7_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['act_reg'])(bn_6)
bn_7 = tf.keras.layers.BatchNormalization()(v_conv_7)

v_conv_8 = tf.keras.layers.Conv3D(filters=params['v_conv_8_filters'], kernel_size=params['v_conv_8_kernel'], strides=params['v_conv_8_strides'], padding=params['v_conv_8_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['act_reg'])(bn_7)
bn_8 = tf.keras.layers.BatchNormalization()(v_conv_8)

v_conv_9 = tf.keras.layers.Conv3D(filters=params['v_conv_9_filters'], kernel_size=params['v_conv_9_kernel'], strides=params['v_conv_9_strides'], padding=params['v_conv_9_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['act_reg'])(bn_8)
bn_9 = tf.keras.layers.BatchNormalization()(v_conv_9)

v_conv_10 = tf.keras.layers.Conv3D(filters=params['v_conv_10_filters'], kernel_size=params['v_conv_10_kernel'], strides=params['v_conv_10_strides'], padding=params['v_conv_10_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['act_reg'])(bn_9)
bn_10 = tf.keras.layers.BatchNormalization()(v_conv_10)


v_flatten_0 = tf.keras.layers.Flatten()(bn_10)

dense_0_vox = tf.keras.layers.Dense(units=params['dense_0_vox_units'], activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['act_reg'])(v_flatten_0)
bn_0 = tf.keras.layers.BatchNormalization()(dense_0_vox)

dense_1_vox = tf.keras.layers.Dense(units=params['dense_1_vox_units'], activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['act_reg'])(bn_0)
bn_1 = tf.keras.layers.BatchNormalization()(dense_1_vox)

dense_2_vox = tf.keras.layers.Dense(units=params['dense_1_vox_units'], activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['act_reg'])(bn_1)
bn_2_v = tf.keras.layers.BatchNormalization()(dense_2_vox)



# -----------------------------------------------------------------
# Fluoro Analysis 1

conv_0 = tf.keras.layers.Conv2D(filters=params['conv_0_filters'], kernel_size=params['conv_0_kernel'], strides=params['conv_0_strides'], padding=params['conv_0_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['act_reg'])(input_fluoro_1)
bn_0 = tf.keras.layers.BatchNormalization()(conv_0)

conv_1 = tf.keras.layers.Conv2D(filters=params['conv_1_filters'], kernel_size=params['conv_1_kernel'], strides=params['conv_1_strides'], padding=params['conv_1_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['act_reg'])(bn_0)
bn_1 = tf.keras.layers.BatchNormalization()(conv_1)

pool_0 = tf.keras.layers.AveragePooling2D(pool_size=params['pool_0_size'], padding=params['pool_0_pad'], data_format=channel_order)(bn_1)

conv_2 = tf.keras.layers.Conv2D(filters=params['conv_2_filters'], kernel_size=params['conv_2_kernel'], strides=params['conv_2_strides'], padding=params['conv_2_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['act_reg'])(pool_0)
bn_2 = tf.keras.layers.BatchNormalization()(conv_2)

conv_3 = tf.keras.layers.Conv2D(filters=params['conv_3_filters'], kernel_size=params['conv_3_kernel'], strides=params['conv_3_strides'], padding=params['conv_3_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['act_reg'])(bn_2)
bn_3 = tf.keras.layers.BatchNormalization()(conv_3)

conv_4 = tf.keras.layers.Conv2D(filters=params['conv_4_filters'], kernel_size=params['conv_4_kernel'], strides=params['conv_4_strides'], padding=params['conv_4_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['act_reg'])(bn_3)
bn_4 = tf.keras.layers.BatchNormalization()(conv_4)

conv_5 = tf.keras.layers.Conv2D(filters=params['conv_5_filters'], kernel_size=params['conv_5_kernel'], strides=params['conv_5_strides'], padding=params['conv_5_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['act_reg'])(bn_4)
bn_5 = tf.keras.layers.BatchNormalization()(conv_5)

conv_6 = tf.keras.layers.Conv2D(filters=params['conv_6_filters'], kernel_size=params['conv_6_kernel'], strides=params['conv_6_strides'], padding=params['conv_6_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['act_reg'])(bn_5)
bn_6 = tf.keras.layers.BatchNormalization()(conv_6)

conv_7 = tf.keras.layers.Conv2D(filters=params['conv_7_filters'], kernel_size=params['conv_7_kernel'], strides=params['conv_7_strides'], padding=params['conv_7_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['act_reg'])(bn_6)
bn_7 = tf.keras.layers.BatchNormalization()(conv_7)

conv_8 = tf.keras.layers.Conv2D(filters=params['conv_8_filters'], kernel_size=params['conv_8_kernel'], strides=params['conv_8_strides'], padding=params['conv_8_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['act_reg'])(bn_7)
bn_8 = tf.keras.layers.BatchNormalization()(conv_8)

flatten_1 = tf.keras.layers.Flatten()(bn_8)

dense_0_flu = tf.keras.layers.Dense(units=params['dense_0_flu_units'], activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['act_reg'])(flatten_1)
bn_0 = tf.keras.layers.BatchNormalization()(dense_0_flu)

dense_1_flu = tf.keras.layers.Dense(units=params['dense_1_flu_units'], activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['act_reg'])(bn_0)
bn_1 = tf.keras.layers.BatchNormalization()(dense_1_flu)

dense_2_flu = tf.keras.layers.Dense(units=params['dense_2_flu_units'], activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['act_reg'])(bn_1)
bn_2_1 = tf.keras.layers.BatchNormalization()(dense_2_flu)

# -----------------------------------------------------------------
# Fluoro Analysis 2

conv_0 = tf.keras.layers.Conv2D(filters=params['conv_0_filters'], kernel_size=params['conv_0_kernel'], strides=params['conv_0_strides'], padding=params['conv_0_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['act_reg'])(input_fluoro_2)
bn_0 = tf.keras.layers.BatchNormalization()(conv_0)

conv_1 = tf.keras.layers.Conv2D(filters=params['conv_1_filters'], kernel_size=params['conv_1_kernel'], strides=params['conv_1_strides'], padding=params['conv_1_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['act_reg'])(bn_0)
bn_1 = tf.keras.layers.BatchNormalization()(conv_1)

pool_0 = tf.keras.layers.AveragePooling2D(pool_size=params['pool_0_size'], padding=params['pool_0_pad'], data_format=channel_order)(bn_1)

conv_2 = tf.keras.layers.Conv2D(filters=params['conv_2_filters'], kernel_size=params['conv_2_kernel'], strides=params['conv_2_strides'], padding=params['conv_2_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['act_reg'])(pool_0)
bn_2 = tf.keras.layers.BatchNormalization()(conv_2)

conv_3 = tf.keras.layers.Conv2D(filters=params['conv_3_filters'], kernel_size=params['conv_3_kernel'], strides=params['conv_3_strides'], padding=params['conv_3_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['act_reg'])(bn_2)
bn_3 = tf.keras.layers.BatchNormalization()(conv_3)

conv_4 = tf.keras.layers.Conv2D(filters=params['conv_4_filters'], kernel_size=params['conv_4_kernel'], strides=params['conv_4_strides'], padding=params['conv_4_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['act_reg'])(bn_3)
bn_4 = tf.keras.layers.BatchNormalization()(conv_4)

conv_5 = tf.keras.layers.Conv2D(filters=params['conv_5_filters'], kernel_size=params['conv_5_kernel'], strides=params['conv_5_strides'], padding=params['conv_5_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['act_reg'])(bn_4)
bn_5 = tf.keras.layers.BatchNormalization()(conv_5)

conv_6 = tf.keras.layers.Conv2D(filters=params['conv_6_filters'], kernel_size=params['conv_6_kernel'], strides=params['conv_6_strides'], padding=params['conv_6_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['act_reg'])(bn_5)
bn_6 = tf.keras.layers.BatchNormalization()(conv_6)

conv_7 = tf.keras.layers.Conv2D(filters=params['conv_7_filters'], kernel_size=params['conv_7_kernel'], strides=params['conv_7_strides'], padding=params['conv_7_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['act_reg'])(bn_6)
bn_7 = tf.keras.layers.BatchNormalization()(conv_7)

conv_8 = tf.keras.layers.Conv2D(filters=params['conv_8_filters'], kernel_size=params['conv_8_kernel'], strides=params['conv_8_strides'], padding=params['conv_8_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['act_reg'])(bn_7)
bn_8 = tf.keras.layers.BatchNormalization()(conv_8)

flatten_1 = tf.keras.layers.Flatten()(bn_8)

dense_0_flu = tf.keras.layers.Dense(units=params['dense_0_flu_units'], activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['act_reg'])(flatten_1)
bn_0 = tf.keras.layers.BatchNormalization()(dense_0_flu)

dense_1_flu = tf.keras.layers.Dense(units=params['dense_1_flu_units'], activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['act_reg'])(bn_0)
bn_1 = tf.keras.layers.BatchNormalization()(dense_1_flu)

dense_2_flu = tf.keras.layers.Dense(units=params['dense_2_flu_units'], activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['act_reg'])(bn_1)
bn_2_2 = tf.keras.layers.BatchNormalization()(dense_2_flu)


# -----------------------------------------------------------------
# Dense After Cali

dense_0_cali = tf.keras.layers.Dense(units=params['dense_0_cali_units'], activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['act_reg'])(input_cali)
bn_0 = tf.keras.layers.BatchNormalization()(dense_0_cali)

dense_1_cali = tf.keras.layers.Dense(units=params['dense_1_cali_units'], activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['act_reg'])(bn_0)
bn_1 = tf.keras.layers.BatchNormalization()(dense_1_cali)

dense_2_cali = tf.keras.layers.Dense(units=params['dense_2_cali_units'], activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['act_reg'])(bn_1)
bn_2 = tf.keras.layers.BatchNormalization()(dense_2_cali)

dense_3_cali = tf.keras.layers.Dense(units=params['dense_3_cali_units'], activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['act_reg'])(bn_2)
bn_3 = tf.keras.layers.BatchNormalization()(dense_3_cali)

dense_4_cali = tf.keras.layers.Dense(units=params['dense_4_cali_units'], activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['act_reg'])(bn_3)
bn_4_cali = tf.keras.layers.BatchNormalization()(dense_4_cali)


# -----------------------------------------------------------------
# Combine Cali, Vox, Fluoro

# concat_1 = tf.keras.layers.concatenate([bn_4_cali])

# concat_1 = tf.keras.layers.concatenate([bn_4_cali, bn_2_1, bn_2_2])

concat_1 = tf.keras.layers.concatenate([bn_4_cali, bn_2_1, bn_2_2, bn_2_v])


# -----------------------------------------------------------------
# Dense Layers at Top

# bn_1 = tf.keras.layers.BatchNormalization()(input_cali)
# bn_1 = tf.keras.layers.BatchNormalization()(input_cali_1)

# dense_1_comb = tf.keras.layers.Dense(units=params['dense_1_co_units'], activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['act_reg'])(bn_4_cali)

dense_1_comb = tf.keras.layers.Dense(units=params['dense_1_co_units'], activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['act_reg'])(concat_1)


bn_2 = tf.keras.layers.BatchNormalization()(dense_1_comb)
dense_2_comb = tf.keras.layers.Dense(units=params['dense_2_co_units'], activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['act_reg'])(bn_2)
bn_3 = tf.keras.layers.BatchNormalization()(dense_2_comb)
dense_3_comb = tf.keras.layers.Dense(units=params['dense_3_co_units'], activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['act_reg'])(bn_3)
bn_4 = tf.keras.layers.BatchNormalization()(dense_3_comb)
dense_4_comb = tf.keras.layers.Dense(units=params['dense_4_co_units'], activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['act_reg'])(bn_4)
bn_5 = tf.keras.layers.BatchNormalization()(dense_4_comb)
dense_5_comb = tf.keras.layers.Dense(units=params['dense_5_co_units'], activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['act_reg'])(bn_5)
bn_6 = tf.keras.layers.BatchNormalization()(dense_5_comb)
dense_6_comb = tf.keras.layers.Dense(units=params['dense_6_co_units'], activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['act_reg'])(bn_6)
bn_7 = tf.keras.layers.BatchNormalization()(dense_6_comb)
dense_7_comb = tf.keras.layers.Dense(units=params['dense_7_co_units'], activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['act_reg'])(bn_7)
bn_8 = tf.keras.layers.BatchNormalization()(dense_7_comb)
dense_8_comb = tf.keras.layers.Dense(units=params['dense_8_co_units'], activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['act_reg'])(bn_8)


# -----------------------------------------------------------------
# Output Layer

main_output = tf.keras.layers.Dense(units=params['main_output_units'], activation=None, kernel_initializer=params['kern_init'], activity_regularizer=None, name='main_output')(dense_8_comb)


# -----------------------------------------------------------------

# Model Housekeeping

model = tf.keras.Model(inputs=[input_vox, input_fluoro_1, input_fluoro_2, input_cali], outputs=main_output)

# model = tf.keras.Model(inputs=[input_fluoro_1, input_fluoro_2, input_cali], outputs=main_output)

run_opts = tf.compat.v1.RunOptions(report_tensor_allocations_upon_oom=True)




model.compile(optimizer=params['model_opt'](lr=params['learning_rate']), loss=params['model_loss'], metrics=[params['model_metric']], options=run_opts)

if save_image:
    tf.keras.utils.plot_model(model, os.path.abspath(os.path.join(py_dir, 'run_' + run_id + '.png')), show_shapes=True)

model.summary()


# -----------------------------------------------------------------

# Next we are going to ensure that we can accurately load sample data


num_of_samples = None


vox_file = h5py.File(os.path.expanduser('~/fluoro/data/compilation/voxels_mark_origin_comp.h5py'), 'r')
vox_mat_base = vox_file['vox_dset']


image_file = h5py.File(os.path.expanduser('~/fluoro/data/compilation/images_norm_std.h5py'), 'r')
image_grp_1 = image_file['image_1']
image_grp_2 = image_file['image_2']
image_init_1 = image_grp_1['min_max_dset_per_image']
image_init_2 = image_grp_2['min_max_dset_per_image']


cali_file = h5py.File(os.path.expanduser('~/fluoro/data/compilation/calibration.h5py'), 'r')
cali_mat_base = cali_file['cali_len3_rot']


label_file = h5py.File(os.path.expanduser('~/fluoro/data/compilation/labels.h5py'), 'r')
label_mat_base = label_file['labels_dset']

# -----------------------------------------------------------------

test_indxs, train_sup_indxs = split_train_test(len(label_mat_base), num_of_samples=num_of_samples)
val_indxs, train_indxs = split_train_test(len(train_sup_indxs))

val_indxs = train_sup_indxs[val_indxs]
train_indxs = train_sup_indxs[train_indxs]

test_indxs = sorted(list(test_indxs))
val_indxs = sorted(list(val_indxs))
train_indxs = sorted(list(train_indxs))


hist_file = open(os.path.join(save_dir, expr_name + '_hist_objects' + '.pkl'), 'wb')

var_dict = {}

var_dict['test_indxs'] = test_indxs
var_dict['val_indxs'] = val_indxs
var_dict['train_indxs'] = train_indxs

var_dict['cali_train_avg'] = np.mean(cali_mat_base[sorted(train_indxs)], axis=0)
var_dict['cali_train_std'] = np.std(cali_mat_base[sorted(train_indxs)], axis=0)
var_dict['cali_train_min'] = np.min(cali_mat_base[sorted(train_indxs)], axis=0)
var_dict['cali_train_max'] = np.max(cali_mat_base[sorted(train_indxs)], axis=0)


var_dict['label_train_avg'] = np.mean(label_mat_base[sorted(train_indxs)], axis=0)
var_dict['label_train_std'] = np.std(label_mat_base[sorted(train_indxs)], axis=0)
var_dict['label_train_min'] = np.min(label_mat_base[sorted(train_indxs)], axis=0)
var_dict['label_train_max'] = np.max(label_mat_base[sorted(train_indxs)], axis=0)


vox_mat_sup = vox_mat_base[:]
vox_mat_train = vox_mat_sup[train_indxs]
vox_mat_val = vox_mat_sup[val_indxs]

image_mat_sup_1 = image_init_1[:]
image_mat_sup_2 = image_init_2[:]
image_mat_train_1 = image_mat_sup_1[train_indxs]
image_mat_train_2 = image_mat_sup_2[train_indxs]
image_mat_val_1 = image_mat_sup_1[val_indxs]
image_mat_val_2 = image_mat_sup_2[val_indxs]


cali_mat_sup = cali_mat_base[:]
cali_train_min_max = min_max_norm(cali_mat_sup[train_indxs])
cali_val_min_max = min_max_norm(cali_mat_sup[val_indxs], data_min=var_dict['cali_train_min'], data_max=var_dict['cali_train_max'])

label_mat_sup = label_mat_base[:]
label_train_min_max = min_max_norm(label_mat_sup[train_indxs])
label_val_min_max = min_max_norm(label_mat_sup[val_indxs], data_min=var_dict['label_train_min'], data_max=var_dict['label_train_max'])


vox_file.close()
image_file.close()
cali_file.close()
label_file.close()


# -----------------------------------------------------------------


tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir, histogram_freq=1, batch_size=params['model_batchsize'], write_grads=True, write_images=True)
terminate_if_nan = tf.keras.callbacks.TerminateOnNaN()




result = model.fit(x={'input_vox': np.expand_dims(vox_mat_train, axis=-1),
                      'input_fluoro_1': image_mat_train_1,
                      'input_fluoro_2': image_mat_train_2,
                      'input_cali': cali_train_min_max},
                   y=label_train_min_max,
                   validation_data=([np.expand_dims(vox_mat_val, axis=-1),
                                     image_mat_val_1,
                                     image_mat_val_2,
                                     cali_val_min_max],
                                    label_val_min_max),
                   epochs=params['model_epochs'],
                   batch_size=params['model_batchsize'],
                   shuffle=True,
                   verbose=2)





model.save(os.path.abspath(os.path.join(save_dir, expr_name + '.h5')))


var_dict['result'] = result.history

pickle.dump(var_dict, hist_file)
hist_file.close()
