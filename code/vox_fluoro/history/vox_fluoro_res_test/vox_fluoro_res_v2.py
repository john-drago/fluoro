import numpy as np
import h5py
import tensorflow as tf
# import keras
import os
import sys
import pickle


# We are going to try to do some residual netowrks


expr_name = sys.argv[0][:-3]
expr_no = '1'
save_dir = os.path.abspath(os.path.join(os.path.expanduser('~/fluoro/code/jupyt/vox_fluoro'), expr_name))
print(save_dir)
os.makedirs(save_dir, exist_ok=True)

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


params = {
    # ---
    # 3D CONV
    # ---
    # Entry Layers
    'v_conv_0_filters': 30,
    'v_conv_0_kernel': 9,
    'v_conv_0_strides_0': 2,
    'v_conv_0_strides_1': 2,
    'v_conv_0_strides_2': 2,
    'v_conv_0_pad': 'same',


    'v_spatial_drop_rate_0': 0.3,

    'v_conv_1_filters': 30,
    'v_conv_1_kernel': 5,
    'v_conv_1_strides_0': 2,
    'v_conv_1_strides_1': 2,
    'v_conv_1_strides_2': 3,
    'v_conv_1_pad': 'same',

    # ---
    # Pool After Initial Layers
    'v_pool_0_size': 2,
    'v_pool_0_pad': 'same',

    # ---
    # Second Run of Entry Layers
    'v_conv_2_filters': 30,
    'v_conv_2_kernel': 5,
    'v_conv_2_strides_0': 2,
    'v_conv_2_strides_1': 2,
    'v_conv_2_strides_2': 2,
    'v_conv_2_pad': 'same',

    # ---
    # Run of Residual Layers

    # 1
    'v_conv_3_filters': 30,
    'v_conv_3_kernel': 3,
    'v_conv_3_strides_0': 1,
    'v_conv_3_strides_1': 1,
    'v_conv_3_strides_2': 1,
    'v_conv_3_pad': 'same',

    'v_spatial_drop_rate_2': 0.3,

    'v_conv_4_filters': 30,
    'v_conv_4_kernel': 3,
    'v_conv_4_strides_0': 1,
    'v_conv_4_strides_1': 1,
    'v_conv_4_strides_2': 1,
    'v_conv_4_pad': 'same',

    # 2
    'v_conv_5_filters': 30,
    'v_conv_5_kernel': 3,
    'v_conv_5_strides_0': 1,
    'v_conv_5_strides_1': 1,
    'v_conv_5_strides_2': 1,
    'v_conv_5_pad': 'same',

    'v_spatial_drop_rate_3': 0.3,

    'v_conv_6_filters': 30,
    'v_conv_6_kernel': 3,
    'v_conv_6_strides_0': 1,
    'v_conv_6_strides_1': 1,
    'v_conv_6_strides_2': 1,
    'v_conv_6_pad': 'same',

    # 3
    'v_conv_7_filters': 30,
    'v_conv_7_kernel': 3,
    'v_conv_7_strides_0': 1,
    'v_conv_7_strides_1': 1,
    'v_conv_7_strides_2': 1,
    'v_conv_7_pad': 'same',

    'v_spatial_drop_rate_4': 0.3,

    'v_conv_8_filters': 30,
    'v_conv_8_kernel': 3,
    'v_conv_8_strides_0': 1,
    'v_conv_8_strides_1': 1,
    'v_conv_8_strides_2': 1,
    'v_conv_8_pad': 'same',

    # 4
    'v_conv_9_filters': 40,
    'v_conv_9_kernel': 3,
    'v_conv_9_strides_0': 2,
    'v_conv_9_strides_1': 2,
    'v_conv_9_strides_2': 2,
    'v_conv_9_pad': 'same',

    'v_spatial_drop_rate_5': 0.3,

    'v_conv_10_filters': 40,
    'v_conv_10_kernel': 3,
    'v_conv_10_strides_0': 1,
    'v_conv_10_strides_1': 1,
    'v_conv_10_strides_2': 1,
    'v_conv_10_pad': 'same',

    'v_conv_11_filters': 40,
    'v_conv_11_kernel': 3,
    'v_conv_11_strides_0': 2,
    'v_conv_11_strides_1': 2,
    'v_conv_11_strides_2': 2,
    'v_conv_11_pad': 'same',

    # 5
    'v_conv_12_filters': 50,
    'v_conv_12_kernel': 2,
    'v_conv_12_strides_0': 2,
    'v_conv_12_strides_1': 2,
    'v_conv_12_strides_2': 2,
    'v_conv_12_pad': 'same',

    'v_spatial_drop_rate_6': 0.3,

    'v_conv_13_filters': 50,
    'v_conv_13_kernel': 2,
    'v_conv_13_strides_0': 1,
    'v_conv_13_strides_1': 1,
    'v_conv_13_strides_2': 1,
    'v_conv_13_pad': 'same',

    'v_conv_14_filters': 50,
    'v_conv_14_kernel': 1,
    'v_conv_14_strides_0': 2,
    'v_conv_14_strides_1': 2,
    'v_conv_14_strides_2': 2,
    'v_conv_14_pad': 'same',

    # 6
    'v_conv_15_filters': 50,
    'v_conv_15_kernel': 2,
    'v_conv_15_strides_0': 2,
    'v_conv_15_strides_1': 2,
    'v_conv_15_strides_2': 2,
    'v_conv_15_pad': 'same',

    'v_spatial_drop_rate_7': 0.3,

    'v_conv_16_filters': 50,
    'v_conv_16_kernel': 2,
    'v_conv_16_strides_0': 1,
    'v_conv_16_strides_1': 1,
    'v_conv_16_strides_2': 1,
    'v_conv_16_pad': 'same',

    'v_conv_17_filters': 50,
    'v_conv_17_kernel': 1,
    'v_conv_17_strides_0': 2,
    'v_conv_17_strides_1': 2,
    'v_conv_17_strides_2': 2,
    'v_conv_17_pad': 'same',

    # ---
    # Final Convs
    'v_spatial_drop_rate_8': 0.5,

    'v_conv_18_filters': 50,
    'v_conv_18_kernel': 2,
    'v_conv_18_strides_0': 1,
    'v_conv_18_strides_1': 1,
    'v_conv_18_strides_2': 1,
    'v_conv_18_pad': 'valid',

    'dense_1_v_units': 75,
    'dense_2_v_units': 50,



    # ---
    # 2D CONV
    # ---
    # Entry Fluoro Layers
    'conv_0_filters': 30,
    'conv_0_kernel': 5,
    'conv_0_strides': 2,
    'conv_0_pad': 'same',

    'spatial_drop_rate_0': 0.3,

    'conv_1_filters': 30,
    'conv_1_kernel': 5,
    'conv_1_strides': 2,
    'conv_1_pad': 'same',

    # ---
    # Pool After Initial Layers
    'pool_0_size': 2,
    'pool_0_pad': 'same',

    # ---
    # Run Of Residual Layers
    # 1
    'conv_2_filters': 30,
    'conv_2_kernel': 3,
    'conv_2_strides': 1,
    'conv_2_pad': 'same',

    'spatial_drop_rate_1': 0.3,

    'conv_3_filters': 30,
    'conv_3_kernel': 3,
    'conv_3_strides': 1,
    'conv_3_pad': 'same',

    # 2
    'conv_4_filters': 30,
    'conv_4_kernel': 3,
    'conv_4_strides': 1,
    'conv_4_pad': 'same',

    'spatial_drop_rate_2': 0.3,

    'conv_5_filters': 30,
    'conv_5_kernel': 3,
    'conv_5_strides': 1,
    'conv_5_pad': 'same',

    # 3
    'conv_6_filters': 30,
    'conv_6_kernel': 3,
    'conv_6_strides': 1,
    'conv_6_pad': 'same',

    'spatial_drop_rate_3': 0.3,

    'conv_7_filters': 30,
    'conv_7_kernel': 3,
    'conv_7_strides': 1,
    'conv_7_pad': 'same',

    # 4
    'conv_8_filters': 30,
    'conv_8_kernel': 3,
    'conv_8_strides': 1,
    'conv_8_pad': 'same',

    'spatial_drop_rate_4': 0.3,

    'conv_9_filters': 30,
    'conv_9_kernel': 3,
    'conv_9_strides': 1,
    'conv_9_pad': 'same',

    # 5
    'conv_10_filters': 40,
    'conv_10_kernel': 3,
    'conv_10_strides': 2,
    'conv_10_pad': 'same',

    'spatial_drop_rate_5': 0.3,

    'conv_11_filters': 40,
    'conv_11_kernel': 3,
    'conv_11_strides': 1,
    'conv_11_pad': 'same',

    'conv_12_filters': 40,
    'conv_12_kernel': 1,
    'conv_12_strides': 2,
    'conv_12_pad': 'same',

    # 6
    'conv_13_filters': 40,
    'conv_13_kernel': 3,
    'conv_13_strides': 2,
    'conv_13_pad': 'same',

    'spatial_drop_rate_6': 0.3,

    'conv_14_filters': 40,
    'conv_14_kernel': 3,
    'conv_14_strides': 1,
    'conv_14_pad': 'same',

    'conv_15_filters': 40,
    'conv_15_kernel': 1,
    'conv_15_strides': 2,
    'conv_15_pad': 'same',

    # 7
    'conv_16_filters': 40,
    'conv_16_kernel': 3,
    'conv_16_strides': 2,
    'conv_16_pad': 'same',

    'spatial_drop_rate_7': 0.3,

    'conv_17_filters': 40,
    'conv_17_kernel': 3,
    'conv_17_strides': 1,
    'conv_17_pad': 'same',

    'conv_18_filters': 40,
    'conv_18_kernel': 1,
    'conv_18_strides': 2,
    'conv_18_pad': 'same',


    # ---
    # Final Conv Layers

    'spatial_drop_rate_8': 0.3,

    'conv_19_filters': 50,
    'conv_19_kernel': 2,
    'conv_19_strides': 1,
    'conv_19_pad': 'valid',


    # ---
    # Dense Layers

    'dense_0_f_units': 50,
    'dense_1_f_units': 50,









    # Calibration Dense Layers
    'dense_1_cali_units': 20,
    'dense_2_cali_units': 20,
    'dense_3_cali_units': 20,

    # Top Level Dense Units
    'dense_1_co_units': 250,
    'drop_1_comb_rate': 0.2,
    'dense_2_co_units': 150,
    'dense_3_co_units': 100,
    'drop_2_comb_rate': 0.2,
    'dense_4_co_units': 20,

    # Main Output
    'main_output_units': 6,
    'main_output_act': 'linear',

    # General Housekeeping
    'v_conv_regularizer': None,
    'conv_regularizer': None,
    'dense_regularizer_1': None,
    'dense_regularizer_2': None,
    'activation_fn': 'elu',
    'kern_init': 'glorot_uniform',
    'model_opt': tf.keras.optimizers.Adam,
    'learning_rate': 0.001,
    'model_epochs': 50,
    'model_batchsize': 5,
    'model_loss': cust_mean_squared_error_var,
    'model_metric': cust_mean_squared_error_var

}

# -----------------------------------------------------------------

channel_order = 'channels_last'
img_input_shape = (128, 128, 1)
vox_input_shape = (199, 164, 566, 1)
cali_input_shape = (6,)


# Input Layers
input_vox = tf.keras.Input(shape=vox_input_shape, name='input_vox', dtype='float32')
input_fluoro_1 = tf.keras.Input(shape=img_input_shape, name='input_fluoro_1', dtype='float32')
input_fluoro_2 = tf.keras.Input(shape=img_input_shape, name='input_fluoro_2', dtype='float32')
input_cali = tf.keras.Input(shape=cali_input_shape, name='input_cali', dtype='float32')

# -----------------------------------------------------------------


# ---
# Entry Layers
v_conv_0 = tf.keras.layers.Conv3D(filters=params['v_conv_0_filters'], kernel_size=params['v_conv_0_kernel'], strides=(params['v_conv_0_strides_0'], params['v_conv_0_strides_1'], params['v_conv_0_strides_2']), padding=params['v_conv_0_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['v_conv_regularizer'])(input_vox)
bn_0 = tf.keras.layers.BatchNormalization()(v_conv_0)


v_spat_0 = tf.keras.layers.SpatialDropout3D(rate=params['v_spatial_drop_rate_0'])(bn_0)
v_conv_1 = tf.keras.layers.Conv3D(filters=params['v_conv_1_filters'], kernel_size=params['v_conv_1_kernel'], strides=(params['v_conv_1_strides_0'], params['v_conv_1_strides_1'], params['v_conv_1_strides_2']), padding=params['v_conv_1_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['v_conv_regularizer'])(v_spat_0)

# ---
# Pool After Initial Layers
v_pool_0 = tf.keras.layers.MaxPooling3D(pool_size=params['v_pool_0_size'], padding=params['v_pool_0_pad'], data_format=channel_order)(v_conv_1)

# ---
# Second Run of Entry Layers
bn_1 = tf.keras.layers.BatchNormalization()(v_pool_0)
v_conv_2 = tf.keras.layers.Conv3D(filters=params['v_conv_2_filters'], kernel_size=params['v_conv_2_kernel'], strides=(params['v_conv_2_strides_0'], params['v_conv_2_strides_1'], params['v_conv_2_strides_2']), padding=params['v_conv_2_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['v_conv_regularizer'])(bn_1)



# ---
# Run of Residual Layers

# 1
bn_2 = tf.keras.layers.BatchNormalization()(v_conv_2)
v_conv_3 = tf.keras.layers.Conv3D(filters=params['v_conv_3_filters'], kernel_size=params['v_conv_3_kernel'], strides=(params['v_conv_3_strides_0'], params['v_conv_3_strides_1'], params['v_conv_3_strides_2']), padding=params['v_conv_3_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['v_conv_regularizer'])(bn_2)
bn_3 = tf.keras.layers.BatchNormalization()(v_conv_3)
v_spat_2 = tf.keras.layers.SpatialDropout3D(rate=params['v_spatial_drop_rate_2'])(bn_3)
v_conv_4 = tf.keras.layers.Conv3D(filters=params['v_conv_4_filters'], kernel_size=params['v_conv_4_kernel'], strides=(params['v_conv_4_strides_0'], params['v_conv_4_strides_1'], params['v_conv_4_strides_2']), padding=params['v_conv_4_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['v_conv_regularizer'])(v_spat_2)
v_add_0 = tf.keras.layers.Add()([v_conv_4, bn_2])

# 2
bn_4 = tf.keras.layers.BatchNormalization()(v_add_0)
v_conv_5 = tf.keras.layers.Conv3D(filters=params['v_conv_5_filters'], kernel_size=params['v_conv_5_kernel'], strides=(params['v_conv_5_strides_0'], params['v_conv_5_strides_1'], params['v_conv_5_strides_2']), padding=params['v_conv_5_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['v_conv_regularizer'])(bn_4)
bn_5 = tf.keras.layers.BatchNormalization()(v_conv_5)
v_spat_3 = tf.keras.layers.SpatialDropout3D(rate=params['v_spatial_drop_rate_3'])(bn_5)
v_conv_6 = tf.keras.layers.Conv3D(filters=params['v_conv_6_filters'], kernel_size=params['v_conv_6_kernel'], strides=(params['v_conv_6_strides_0'], params['v_conv_6_strides_1'], params['v_conv_6_strides_2']), padding=params['v_conv_6_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['v_conv_regularizer'])(v_spat_3)
v_add_1 = tf.keras.layers.Add()([v_conv_6, bn_4])

# 3
bn_6 = tf.keras.layers.BatchNormalization()(v_add_1)
v_conv_7 = tf.keras.layers.Conv3D(filters=params['v_conv_7_filters'], kernel_size=params['v_conv_7_kernel'], strides=(params['v_conv_7_strides_0'], params['v_conv_7_strides_1'], params['v_conv_7_strides_2']), padding=params['v_conv_7_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['v_conv_regularizer'])(bn_6)
bn_7 = tf.keras.layers.BatchNormalization()(v_conv_7)
v_spat_4 = tf.keras.layers.SpatialDropout3D(rate=params['v_spatial_drop_rate_4'])(bn_7)
v_conv_8 = tf.keras.layers.Conv3D(filters=params['v_conv_8_filters'], kernel_size=params['v_conv_8_kernel'], strides=(params['v_conv_8_strides_0'], params['v_conv_8_strides_1'], params['v_conv_8_strides_2']), padding=params['v_conv_8_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['v_conv_regularizer'])(v_spat_4)
v_add_2 = tf.keras.layers.Add()([v_conv_8, bn_6])

# 4
bn_8 = tf.keras.layers.BatchNormalization()(v_add_2)
v_conv_9 = tf.keras.layers.Conv3D(filters=params['v_conv_9_filters'], kernel_size=params['v_conv_9_kernel'], strides=(params['v_conv_9_strides_0'], params['v_conv_9_strides_1'], params['v_conv_9_strides_2']), padding=params['v_conv_9_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['v_conv_regularizer'])(bn_8)
bn_9 = tf.keras.layers.BatchNormalization()(v_conv_9)
v_spat_5 = tf.keras.layers.SpatialDropout3D(rate=params['v_spatial_drop_rate_5'])(bn_9)
v_conv_10 = tf.keras.layers.Conv3D(filters=params['v_conv_10_filters'], kernel_size=params['v_conv_10_kernel'], strides=(params['v_conv_10_strides_0'], params['v_conv_10_strides_1'], params['v_conv_10_strides_2']), padding=params['v_conv_10_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['v_conv_regularizer'])(v_spat_5)
v_conv_11 = tf.keras.layers.Conv3D(filters=params['v_conv_11_filters'], kernel_size=params['v_conv_11_kernel'], strides=(params['v_conv_11_strides_0'], params['v_conv_11_strides_1'], params['v_conv_11_strides_2']), padding=params['v_conv_11_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['v_conv_regularizer'])(bn_8)
v_add_3 = tf.keras.layers.Add()([v_conv_10, v_conv_11])

# 5
bn_10 = tf.keras.layers.BatchNormalization()(v_add_3)
v_conv_12 = tf.keras.layers.Conv3D(filters=params['v_conv_12_filters'], kernel_size=params['v_conv_12_kernel'], strides=(params['v_conv_12_strides_0'], params['v_conv_12_strides_1'], params['v_conv_12_strides_2']), padding=params['v_conv_12_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['v_conv_regularizer'])(bn_10)
bn_11 = tf.keras.layers.BatchNormalization()(v_conv_12)
v_spat_6 = tf.keras.layers.SpatialDropout3D(rate=params['v_spatial_drop_rate_6'])(bn_11)
v_conv_13 = tf.keras.layers.Conv3D(filters=params['v_conv_13_filters'], kernel_size=params['v_conv_13_kernel'], strides=(params['v_conv_13_strides_0'], params['v_conv_13_strides_1'], params['v_conv_13_strides_2']), padding=params['v_conv_13_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['v_conv_regularizer'])(v_spat_6)
v_conv_14 = tf.keras.layers.Conv3D(filters=params['v_conv_14_filters'], kernel_size=params['v_conv_14_kernel'], strides=(params['v_conv_14_strides_0'], params['v_conv_14_strides_1'], params['v_conv_14_strides_2']), padding=params['v_conv_14_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['v_conv_regularizer'])(bn_10)
v_add_4 = tf.keras.layers.Add()([v_conv_13, v_conv_14])

# 6
bn_12 = tf.keras.layers.BatchNormalization()(v_add_4)
v_conv_15 = tf.keras.layers.Conv3D(filters=params['v_conv_12_filters'], kernel_size=params['v_conv_12_kernel'], strides=(params['v_conv_12_strides_0'], params['v_conv_12_strides_1'], params['v_conv_12_strides_2']), padding=params['v_conv_12_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['v_conv_regularizer'])(bn_12)
bn_13 = tf.keras.layers.BatchNormalization()(v_conv_15)
v_spat_7 = tf.keras.layers.SpatialDropout3D(rate=params['v_spatial_drop_rate_7'])(bn_13)
v_conv_16 = tf.keras.layers.Conv3D(filters=params['v_conv_13_filters'], kernel_size=params['v_conv_13_kernel'], strides=(params['v_conv_13_strides_0'], params['v_conv_13_strides_1'], params['v_conv_13_strides_2']), padding=params['v_conv_13_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['v_conv_regularizer'])(v_spat_7)
v_conv_17 = tf.keras.layers.Conv3D(filters=params['v_conv_14_filters'], kernel_size=params['v_conv_14_kernel'], strides=(params['v_conv_14_strides_0'], params['v_conv_14_strides_1'], params['v_conv_14_strides_2']), padding=params['v_conv_14_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['v_conv_regularizer'])(bn_12)
v_add_5 = tf.keras.layers.Add()([v_conv_16, v_conv_17])

# ---
# Final Conv Layers
bn_14 = tf.keras.layers.BatchNormalization()(v_add_5)
v_spat_8 = tf.keras.layers.SpatialDropout3D(rate=params['v_spatial_drop_rate_8'])(bn_14)
v_conv_18 = tf.keras.layers.Conv3D(filters=params['v_conv_18_filters'], kernel_size=params['v_conv_18_kernel'], strides=(params['v_conv_18_strides_0'], params['v_conv_18_strides_1'], params['v_conv_18_strides_2']), padding=params['v_conv_18_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['v_conv_regularizer'])(v_spat_8)

# ---
# Dense Layers
v_flatten_0 = tf.keras.layers.Flatten()(v_conv_18)

bn_15 = tf.keras.layers.BatchNormalization()(v_flatten_0)
dense_1_v = tf.keras.layers.Dense(units=params['dense_1_v_units'], activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['dense_regularizer_1'])(bn_15)

bn_16 = tf.keras.layers.BatchNormalization()(dense_1_v)
dense_2_v = tf.keras.layers.Dense(units=params['dense_2_v_units'], activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['dense_regularizer_1'])(bn_16)


# -----------------------------------------------------------------

# ---
# Entry Fluoro Layers
per_image_stand_1 = tf.keras.layers.Lambda(lambda frame: tf.image.per_image_standardization(frame))(input_fluoro_1)

bn_0 = tf.keras.layers.BatchNormalization()(per_image_stand_1)
conv_0_1 = tf.keras.layers.Conv2D(filters=params['conv_0_filters'], kernel_size=params['conv_0_kernel'], strides=params['conv_0_strides'], padding=params['conv_0_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['conv_regularizer'])(bn_0)

bn_1 = tf.keras.layers.BatchNormalization()(conv_0_1)
spat_0_1 = tf.keras.layers.SpatialDropout2D(rate=params['spatial_drop_rate_0'])(bn_1)
conv_1_1 = tf.keras.layers.Conv2D(filters=params['conv_1_filters'], kernel_size=params['conv_1_kernel'], strides=params['conv_1_strides'], padding=params['conv_1_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['conv_regularizer'])(spat_0_1)

# ---
# Pool After Initial Layers
pool_0_1 = tf.keras.layers.AveragePooling2D(pool_size=params['pool_0_size'], padding=params['pool_0_pad'])(conv_1_1)

# ---
# Run of Residual Layers
# 1
bn_2 = tf.keras.layers.BatchNormalization()(pool_0_1)
conv_2_1 = tf.keras.layers.Conv2D(filters=params['conv_2_filters'], kernel_size=params['conv_2_kernel'], strides=params['conv_2_strides'], padding=params['conv_2_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['conv_regularizer'])(bn_2)
bn_3 = tf.keras.layers.BatchNormalization()(conv_2_1)
spat_1_1 = tf.keras.layers.SpatialDropout2D(rate=params['spatial_drop_rate_1'])(bn_3)
conv_3_1 = tf.keras.layers.Conv2D(filters=params['conv_3_filters'], kernel_size=params['conv_3_kernel'], strides=params['conv_3_strides'], padding=params['conv_3_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['conv_regularizer'])(spat_1_1)
add_0 = tf.keras.layers.Add()([conv_3_1, bn_2])

# 2
bn_4 = tf.keras.layers.BatchNormalization()(add_0)
conv_4_1 = tf.keras.layers.Conv2D(filters=params['conv_4_filters'], kernel_size=params['conv_4_kernel'], strides=params['conv_4_strides'], padding=params['conv_4_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['conv_regularizer'])(bn_4)
bn_5 = tf.keras.layers.BatchNormalization()(conv_4_1)
spat_2_1 = tf.keras.layers.SpatialDropout2D(rate=params['spatial_drop_rate_2'])(bn_5)
conv_5_1 = tf.keras.layers.Conv2D(filters=params['conv_5_filters'], kernel_size=params['conv_5_kernel'], strides=params['conv_5_strides'], padding=params['conv_5_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['conv_regularizer'])(spat_2_1)
add_1 = tf.keras.layers.Add()([conv_5_1, bn_4])

# 3
bn_6 = tf.keras.layers.BatchNormalization()(add_1)
conv_6_1 = tf.keras.layers.Conv2D(filters=params['conv_6_filters'], kernel_size=params['conv_6_kernel'], strides=params['conv_6_strides'], padding=params['conv_6_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['conv_regularizer'])(bn_6)
bn_7 = tf.keras.layers.BatchNormalization()(conv_6_1)
spat_3_1 = tf.keras.layers.SpatialDropout2D(rate=params['spatial_drop_rate_3'])(bn_7)
conv_7_1 = tf.keras.layers.Conv2D(filters=params['conv_7_filters'], kernel_size=params['conv_7_kernel'], strides=params['conv_7_strides'], padding=params['conv_7_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['conv_regularizer'])(spat_3_1)
add_2 = tf.keras.layers.Add()([conv_7_1, bn_6])

# 4
bn_8 = tf.keras.layers.BatchNormalization()(add_2)
conv_8_1 = tf.keras.layers.Conv2D(filters=params['conv_8_filters'], kernel_size=params['conv_8_kernel'], strides=params['conv_8_strides'], padding=params['conv_8_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['conv_regularizer'])(bn_8)
bn_9 = tf.keras.layers.BatchNormalization()(conv_8_1)
spat_4_1 = tf.keras.layers.SpatialDropout2D(rate=params['spatial_drop_rate_4'])(bn_9)
conv_9_1 = tf.keras.layers.Conv2D(filters=params['conv_9_filters'], kernel_size=params['conv_9_kernel'], strides=params['conv_9_strides'], padding=params['conv_9_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['conv_regularizer'])(spat_4_1)
add_3 = tf.keras.layers.Add()([conv_9_1, bn_8])

# 5
bn_10 = tf.keras.layers.BatchNormalization()(add_3)
conv_10_1 = tf.keras.layers.Conv2D(filters=params['conv_10_filters'], kernel_size=params['conv_10_kernel'], strides=params['conv_10_strides'], padding=params['conv_10_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['conv_regularizer'])(bn_10)
bn_11 = tf.keras.layers.BatchNormalization()(conv_10_1)
spat_5_1 = tf.keras.layers.SpatialDropout2D(rate=params['spatial_drop_rate_5'])(bn_11)
conv_11_1 = tf.keras.layers.Conv2D(filters=params['conv_11_filters'], kernel_size=params['conv_11_kernel'], strides=params['conv_11_strides'], padding=params['conv_11_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['conv_regularizer'])(spat_5_1)
conv_12_1 = tf.keras.layers.Conv2D(filters=params['conv_12_filters'], kernel_size=params['conv_12_kernel'], strides=params['conv_12_strides'], padding=params['conv_12_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['conv_regularizer'])(bn_10)
add_4 = tf.keras.layers.Add()([conv_11_1, conv_12_1])

# 6
bn_12 = tf.keras.layers.BatchNormalization()(add_4)
conv_13_1 = tf.keras.layers.Conv2D(filters=params['conv_13_filters'], kernel_size=params['conv_13_kernel'], strides=params['conv_13_strides'], padding=params['conv_13_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['conv_regularizer'])(bn_12)
bn_13 = tf.keras.layers.BatchNormalization()(conv_13_1)
spat_6_1 = tf.keras.layers.SpatialDropout2D(rate=params['spatial_drop_rate_6'])(bn_13)
conv_14_1 = tf.keras.layers.Conv2D(filters=params['conv_14_filters'], kernel_size=params['conv_14_kernel'], strides=params['conv_14_strides'], padding=params['conv_14_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['conv_regularizer'])(spat_6_1)
conv_15_1 = tf.keras.layers.Conv2D(filters=params['conv_15_filters'], kernel_size=params['conv_15_kernel'], strides=params['conv_15_strides'], padding=params['conv_15_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['conv_regularizer'])(bn_12)
add_5 = tf.keras.layers.Add()([conv_14_1, conv_15_1])

# 7
bn_14 = tf.keras.layers.BatchNormalization()(add_5)
conv_16_1 = tf.keras.layers.Conv2D(filters=params['conv_16_filters'], kernel_size=params['conv_16_kernel'], strides=params['conv_16_strides'], padding=params['conv_16_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['conv_regularizer'])(bn_14)
bn_15 = tf.keras.layers.BatchNormalization()(conv_16_1)
spat_7_1 = tf.keras.layers.SpatialDropout2D(rate=params['spatial_drop_rate_7'])(bn_15)
conv_17_1 = tf.keras.layers.Conv2D(filters=params['conv_17_filters'], kernel_size=params['conv_17_kernel'], strides=params['conv_17_strides'], padding=params['conv_17_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['conv_regularizer'])(spat_7_1)
conv_18_1 = tf.keras.layers.Conv2D(filters=params['conv_18_filters'], kernel_size=params['conv_18_kernel'], strides=params['conv_18_strides'], padding=params['conv_18_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['conv_regularizer'])(bn_14)
add_6 = tf.keras.layers.Add()([conv_17_1, conv_18_1])

# ---
# Final Conv Layers

bn_16 = tf.keras.layers.BatchNormalization()(add_6)
spat_8_1 = tf.keras.layers.SpatialDropout2D(rate=params['spatial_drop_rate_8'])(bn_16)
conv_19_1 = tf.keras.layers.Conv2D(filters=params['conv_19_filters'], kernel_size=params['conv_19_kernel'], strides=params['conv_19_strides'], padding=params['conv_19_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['conv_regularizer'])(spat_8_1)

# ---
# Dense Layers
flatten_0 = tf.keras.layers.Flatten()(conv_19_1)

bn_17 = tf.keras.layers.BatchNormalization()(flatten_0)
dense_0_f_1 = tf.keras.layers.Dense(units=params['dense_0_f_units'], activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['dense_regularizer_1'])(bn_17)

bn_18 = tf.keras.layers.BatchNormalization()(dense_0_f_1)
dense_1_f_1 = tf.keras.layers.Dense(units=params['dense_1_f_units'], activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['dense_regularizer_1'])(bn_18)


# -----------------------------------------------------------------

# ---
# Entry Fluoro Layers
per_image_stand_2 = tf.keras.layers.Lambda(lambda frame: tf.image.per_image_standardization(frame))(input_fluoro_2)

bn_0 = tf.keras.layers.BatchNormalization()(per_image_stand_2)
conv_0_2 = tf.keras.layers.Conv2D(filters=params['conv_0_filters'], kernel_size=params['conv_0_kernel'], strides=params['conv_0_strides'], padding=params['conv_0_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['conv_regularizer'])(bn_0)

bn_1 = tf.keras.layers.BatchNormalization()(conv_0_2)
spat_0_2 = tf.keras.layers.SpatialDropout2D(rate=params['spatial_drop_rate_0'])(bn_1)
conv_1_2 = tf.keras.layers.Conv2D(filters=params['conv_1_filters'], kernel_size=params['conv_1_kernel'], strides=params['conv_1_strides'], padding=params['conv_1_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['conv_regularizer'])(spat_0_2)

# ---
# Pool After Initial Layers
pool_0_2 = tf.keras.layers.AveragePooling2D(pool_size=params['pool_0_size'], padding=params['pool_0_pad'])(conv_1_2)

# ---
# Run of Residual Layers
# 1
bn_2 = tf.keras.layers.BatchNormalization()(pool_0_2)
conv_2_2 = tf.keras.layers.Conv2D(filters=params['conv_2_filters'], kernel_size=params['conv_2_kernel'], strides=params['conv_2_strides'], padding=params['conv_2_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['conv_regularizer'])(bn_2)
bn_3 = tf.keras.layers.BatchNormalization()(conv_2_2)
spat_1_2 = tf.keras.layers.SpatialDropout2D(rate=params['spatial_drop_rate_1'])(bn_3)
conv_3_2 = tf.keras.layers.Conv2D(filters=params['conv_3_filters'], kernel_size=params['conv_3_kernel'], strides=params['conv_3_strides'], padding=params['conv_3_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['conv_regularizer'])(spat_1_2)
add_0 = tf.keras.layers.Add()([conv_3_2, bn_2])

# 2
bn_4 = tf.keras.layers.BatchNormalization()(add_0)
conv_4_2 = tf.keras.layers.Conv2D(filters=params['conv_4_filters'], kernel_size=params['conv_4_kernel'], strides=params['conv_4_strides'], padding=params['conv_4_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['conv_regularizer'])(bn_4)
bn_5 = tf.keras.layers.BatchNormalization()(conv_4_2)
spat_2_2 = tf.keras.layers.SpatialDropout2D(rate=params['spatial_drop_rate_2'])(bn_5)
conv_5_2 = tf.keras.layers.Conv2D(filters=params['conv_5_filters'], kernel_size=params['conv_5_kernel'], strides=params['conv_5_strides'], padding=params['conv_5_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['conv_regularizer'])(spat_2_2)
add_1 = tf.keras.layers.Add()([conv_5_2, bn_4])

# 3
bn_6 = tf.keras.layers.BatchNormalization()(add_1)
conv_6_2 = tf.keras.layers.Conv2D(filters=params['conv_6_filters'], kernel_size=params['conv_6_kernel'], strides=params['conv_6_strides'], padding=params['conv_6_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['conv_regularizer'])(bn_6)
bn_7 = tf.keras.layers.BatchNormalization()(conv_6_2)
spat_3_2 = tf.keras.layers.SpatialDropout2D(rate=params['spatial_drop_rate_3'])(bn_7)
conv_7_2 = tf.keras.layers.Conv2D(filters=params['conv_7_filters'], kernel_size=params['conv_7_kernel'], strides=params['conv_7_strides'], padding=params['conv_7_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['conv_regularizer'])(spat_3_2)
add_2 = tf.keras.layers.Add()([conv_7_2, bn_6])

# 4
bn_8 = tf.keras.layers.BatchNormalization()(add_2)
conv_8_2 = tf.keras.layers.Conv2D(filters=params['conv_8_filters'], kernel_size=params['conv_8_kernel'], strides=params['conv_8_strides'], padding=params['conv_8_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['conv_regularizer'])(bn_8)
bn_9 = tf.keras.layers.BatchNormalization()(conv_8_2)
spat_4_2 = tf.keras.layers.SpatialDropout2D(rate=params['spatial_drop_rate_4'])(bn_9)
conv_9_2 = tf.keras.layers.Conv2D(filters=params['conv_9_filters'], kernel_size=params['conv_9_kernel'], strides=params['conv_9_strides'], padding=params['conv_9_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['conv_regularizer'])(spat_4_2)
add_3 = tf.keras.layers.Add()([conv_9_2, bn_8])

# 5
bn_10 = tf.keras.layers.BatchNormalization()(add_3)
conv_10_2 = tf.keras.layers.Conv2D(filters=params['conv_10_filters'], kernel_size=params['conv_10_kernel'], strides=params['conv_10_strides'], padding=params['conv_10_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['conv_regularizer'])(bn_10)
bn_11 = tf.keras.layers.BatchNormalization()(conv_10_2)
spat_5_2 = tf.keras.layers.SpatialDropout2D(rate=params['spatial_drop_rate_5'])(bn_11)
conv_11_2 = tf.keras.layers.Conv2D(filters=params['conv_11_filters'], kernel_size=params['conv_11_kernel'], strides=params['conv_11_strides'], padding=params['conv_11_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['conv_regularizer'])(spat_5_2)
conv_12_2 = tf.keras.layers.Conv2D(filters=params['conv_12_filters'], kernel_size=params['conv_12_kernel'], strides=params['conv_12_strides'], padding=params['conv_12_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['conv_regularizer'])(bn_10)
add_4 = tf.keras.layers.Add()([conv_11_2, conv_12_2])

# 6
bn_12 = tf.keras.layers.BatchNormalization()(add_4)
conv_13_2 = tf.keras.layers.Conv2D(filters=params['conv_13_filters'], kernel_size=params['conv_13_kernel'], strides=params['conv_13_strides'], padding=params['conv_13_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['conv_regularizer'])(bn_12)
bn_13 = tf.keras.layers.BatchNormalization()(conv_13_2)
spat_6_2 = tf.keras.layers.SpatialDropout2D(rate=params['spatial_drop_rate_6'])(bn_13)
conv_14_2 = tf.keras.layers.Conv2D(filters=params['conv_14_filters'], kernel_size=params['conv_14_kernel'], strides=params['conv_14_strides'], padding=params['conv_14_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['conv_regularizer'])(spat_6_2)
conv_15_2 = tf.keras.layers.Conv2D(filters=params['conv_15_filters'], kernel_size=params['conv_15_kernel'], strides=params['conv_15_strides'], padding=params['conv_15_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['conv_regularizer'])(bn_12)
add_5 = tf.keras.layers.Add()([conv_14_2, conv_15_2])

# 7
bn_14 = tf.keras.layers.BatchNormalization()(add_5)
conv_16_2 = tf.keras.layers.Conv2D(filters=params['conv_16_filters'], kernel_size=params['conv_16_kernel'], strides=params['conv_16_strides'], padding=params['conv_16_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['conv_regularizer'])(bn_14)
bn_15 = tf.keras.layers.BatchNormalization()(conv_16_2)
spat_7_2 = tf.keras.layers.SpatialDropout2D(rate=params['spatial_drop_rate_7'])(bn_15)
conv_17_2 = tf.keras.layers.Conv2D(filters=params['conv_17_filters'], kernel_size=params['conv_17_kernel'], strides=params['conv_17_strides'], padding=params['conv_17_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['conv_regularizer'])(spat_7_2)
conv_18_2 = tf.keras.layers.Conv2D(filters=params['conv_18_filters'], kernel_size=params['conv_18_kernel'], strides=params['conv_18_strides'], padding=params['conv_18_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['conv_regularizer'])(bn_14)
add_6 = tf.keras.layers.Add()([conv_17_2, conv_18_2])

# ---
# Final Conv Layers

bn_16 = tf.keras.layers.BatchNormalization()(add_6)
spat_8_2 = tf.keras.layers.SpatialDropout2D(rate=params['spatial_drop_rate_8'])(bn_16)
conv_19_2 = tf.keras.layers.Conv2D(filters=params['conv_19_filters'], kernel_size=params['conv_19_kernel'], strides=params['conv_19_strides'], padding=params['conv_19_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['conv_regularizer'])(spat_8_2)

# ---
# Dense Layers
flatten_0 = tf.keras.layers.Flatten()(conv_19_2)

bn_17 = tf.keras.layers.BatchNormalization()(flatten_0)
dense_0_f_2 = tf.keras.layers.Dense(units=params['dense_0_f_units'], activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['dense_regularizer_1'])(bn_17)

bn_18 = tf.keras.layers.BatchNormalization()(dense_0_f_2)
dense_1_f_2 = tf.keras.layers.Dense(units=params['dense_1_f_units'], activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['dense_regularizer_1'])(bn_18)

# -----------------------------------------------------------------

# ---
# Combine the fluoro inputs



# -----------------------------------------------------------------



# -----------------------------------------------------------------



# -----------------------------------------------------------------

# Main Output
main_output = tf.keras.layers.Dense(units=params['main_output_units'], activation=params['main_output_act'], kernel_initializer=params['kern_init'], name='main_output')(dense_0_comb)

# -----------------------------------------------------------------

# Model Housekeeping
# model = tf.keras.Model(inputs=[input_vox, input_fluoro_1, input_fluoro_2, input_cali], outputs=main_output)
model = tf.keras.Model(inputs=[input_vox, input_fluoro_1, input_fluoro_2], outputs=main_output)

model.compile(optimizer=params['model_opt'](lr=params['learning_rate']), loss=params['model_loss'], metrics=[params['model_metric']])
tf.keras.utils.plot_model(model, os.path.abspath(os.path.join(save_dir, expr_name + '_' + expr_no + '.png')), show_shapes=True)

model.summary()

# -----------------------------------------------------------------

# vox_file = h5py.File(os.path.expanduser('~/fluoro/data/compilation/voxels_pad.h5py'), 'r')
# vox_init = vox_file['vox_dset']

# image_file = h5py.File(os.path.expanduser('~/fluoro/data/compilation/images.h5py'), 'r')
# image_init = image_file['image_dset']

# label_file = h5py.File(os.path.expanduser('~/fluoro/data/compilation/labels.h5py'), 'r')
# label_init = label_file['labels_dset']

# cali_file = h5py.File(os.path.expanduser('~/fluoro/data/compilation/calibration.h5py'), 'r')
# cali_init = cali_file['cali_len3_rot']


# def split_train_test(shape, num_of_samples=None, ratio=0.2):

#     if num_of_samples is None:
#         shuffled_indices = np.random.choice(shape, size=shape, replace=False)
#     else:
#         shuffled_indices = np.random.choice(shape, size=num_of_samples, replace=False)


#     test_set_size = int(len(shuffled_indices) * 0.2)
#     test_indx = shuffled_indices[:test_set_size]
#     train_indx = shuffled_indices[test_set_size:]

#     return test_indx, train_indx


# num_of_samples = None


# test_indxs, train_sup_indxs = split_train_test(len(label_init), num_of_samples=num_of_samples)
# val_indxs, train_indxs = split_train_test(len(train_sup_indxs))

# val_indxs = train_sup_indxs[val_indxs]
# train_indxs = train_sup_indxs[train_indxs]

# test_indxs = sorted(list(test_indxs))
# val_indxs = sorted(list(val_indxs))
# train_indxs = sorted(list(train_indxs))


# hist_file = open(os.path.join(save_dir, expr_name + '_hist_objects_' + expr_no + '.pkl'), 'wb')

# var_dict = {}

# var_dict['test_indxs'] = test_indxs
# var_dict['val_indxs'] = val_indxs
# var_dict['train_indxs'] = train_indxs

# vox_mat_train = vox_init[:]
# vox_mat_val = vox_mat_train[val_indxs]
# vox_mat_train = vox_mat_train[train_indxs]
# vox_file.close()

# image_mat_train = image_init[:]
# image_mat_val = image_mat_train[val_indxs]
# image_mat_train = image_mat_train[train_indxs]
# image_file.close()

# cali_mat_train = cali_init[:]
# cali_mat_val = cali_mat_train[val_indxs]
# cali_mat_train = cali_mat_train[train_indxs]
# cali_file.close()

# label_mat_train = label_init[:]
# label_mat_val = label_mat_train[val_indxs]
# label_mat_train = label_mat_train[train_indxs]
# label_file.close()

# # -----------------------------------------------------------------


# print('\n\ncompletely loaded...\n\n')


# result = model.fit(x={'input_vox': np.expand_dims(vox_mat_train, axis=-1), 'input_fluoro_1': np.expand_dims(image_mat_train[:, 0, :, :], axis=-1), 'input_fluoro_2': np.expand_dims(image_mat_train[:, 1, :, :], axis=-1), 'input_cali': cali_mat_train}, y=label_mat_train, validation_data=([np.expand_dims(vox_mat_val, axis=-1), np.expand_dims(image_mat_val[:, 0, :, :], axis=-1), np.expand_dims(image_mat_val[:, 1, :, :], axis=-1), cali_mat_val], label_mat_val), epochs=params['model_epochs'], batch_size=params['model_batchsize'], shuffle=True, verbose=2)

# model.save(os.path.abspath(os.path.join(save_dir, expr_name + '_' + expr_no + '.h5')))


# var_dict['result'] = result.history
# pickle.dump(var_dict, hist_file)
# hist_file.close()





