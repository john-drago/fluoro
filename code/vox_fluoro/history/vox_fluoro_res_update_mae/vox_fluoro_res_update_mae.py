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

    'v_intra_act_fn': None,
    'v_res_act_fn': 'elu',

    'v_conv_0_filters': 30,
    'v_conv_0_kernel': 11,
    'v_conv_0_strides_0': 2,
    'v_conv_0_strides_1': 2,
    'v_conv_0_strides_2': 2,
    'v_conv_0_pad': 'same',


    'v_spatial_drop_rate_0': 0.4,

    'v_conv_1_filters': 30,
    'v_conv_1_kernel': 7,
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

    'v_spatial_drop_rate_2': 0.4,

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

    'v_spatial_drop_rate_3': 0.4,

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

    'v_spatial_drop_rate_4': 0.4,

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

    'v_spatial_drop_rate_5': 0.,

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

    'v_spatial_drop_rate_6': 0.4,

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

    'v_spatial_drop_rate_7': 0.4,

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
    'v_spatial_drop_rate_8': 0.4,

    'v_conv_18_filters': 50,
    'v_conv_18_kernel': 2,
    'v_conv_18_strides_0': 1,
    'v_conv_18_strides_1': 1,
    'v_conv_18_strides_2': 1,
    'v_conv_18_pad': 'valid',

    'drop_1_v_rate': 0.3,
    'dense_1_v_units': 75,
    'drop_2_v_rate': 0.3,
    'dense_2_v_units': 50,


    # ---
    # 2D CONV
    # ---

    'intra_act_fn': None,
    'res_act_fn': 'elu',

    # Entry Fluoro Layers
    'conv_0_filters': 30,
    'conv_0_kernel': 5,
    'conv_0_strides': 2,
    'conv_0_pad': 'same',

    'spatial_drop_rate_0': 0.4,

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

    'spatial_drop_rate_1': 0.4,

    'conv_3_filters': 30,
    'conv_3_kernel': 3,
    'conv_3_strides': 1,
    'conv_3_pad': 'same',

    # 2
    'conv_4_filters': 30,
    'conv_4_kernel': 3,
    'conv_4_strides': 1,
    'conv_4_pad': 'same',

    'spatial_drop_rate_2': 0.4,

    'conv_5_filters': 30,
    'conv_5_kernel': 3,
    'conv_5_strides': 1,
    'conv_5_pad': 'same',

    # 3
    'conv_6_filters': 30,
    'conv_6_kernel': 3,
    'conv_6_strides': 1,
    'conv_6_pad': 'same',

    'spatial_drop_rate_3': 0.4,

    'conv_7_filters': 30,
    'conv_7_kernel': 3,
    'conv_7_strides': 1,
    'conv_7_pad': 'same',

    # 4
    'conv_8_filters': 30,
    'conv_8_kernel': 3,
    'conv_8_strides': 1,
    'conv_8_pad': 'same',

    'spatial_drop_rate_4': 0.4,

    'conv_9_filters': 30,
    'conv_9_kernel': 3,
    'conv_9_strides': 1,
    'conv_9_pad': 'same',

    # 5
    'conv_10_filters': 30,
    'conv_10_kernel': 3,
    'conv_10_strides': 1,
    'conv_10_pad': 'same',

    'spatial_drop_rate_5': 0.4,

    'conv_11_filters': 30,
    'conv_11_kernel': 3,
    'conv_11_strides': 1,
    'conv_11_pad': 'same',

    # 6
    'conv_12_filters': 30,
    'conv_12_kernel': 3,
    'conv_12_strides': 1,
    'conv_12_pad': 'same',

    'spatial_drop_rate_6': 0.4,

    'conv_13_filters': 30,
    'conv_13_kernel': 3,
    'conv_13_strides': 1,
    'conv_13_pad': 'same',


    # ---
    # COMB FLUOROS
    # ---

    # ---
    # RES NET AFTER COMB FLUORO
    # ---

    'c_intra_act_fn': None,
    'c_res_act_fn': 'elu',

    # 0
    'comb_0_filters': 60,
    'comb_0_kernel': 3,
    'comb_0_strides': 1,
    'comb_0_pad': 'same',

    'comb_spatial_0': 0.4,

    'comb_1_filters': 60,
    'comb_1_kernel': 3,
    'comb_1_strides': 1,
    'comb_1_pad': 'same',

    # 1
    'comb_2_filters': 60,
    'comb_2_kernel': 3,
    'comb_2_strides': 1,
    'comb_2_pad': 'same',

    'comb_spatial_1': 0.4,

    'comb_3_filters': 60,
    'comb_3_kernel': 3,
    'comb_3_strides': 1,
    'comb_3_pad': 'same',

    # 2
    'comb_4_filters': 60,
    'comb_4_kernel': 3,
    'comb_4_strides': 1,
    'comb_4_pad': 'same',

    'comb_spatial_2': 0.4,

    'comb_5_filters': 60,
    'comb_5_kernel': 3,
    'comb_5_strides': 1,
    'comb_5_pad': 'same',

    # 3
    'comb_6_filters': 60,
    'comb_6_kernel': 3,
    'comb_6_strides': 1,
    'comb_6_pad': 'same',

    'comb_spatial_3': 0.4,

    'comb_7_filters': 60,
    'comb_7_kernel': 3,
    'comb_7_strides': 1,
    'comb_7_pad': 'same',

    # 4
    'comb_8_filters': 60,
    'comb_8_kernel': 3,
    'comb_8_strides': 1,
    'comb_8_pad': 'same',

    'comb_spatial_4': 0.4,

    'comb_9_filters': 60,
    'comb_9_kernel': 3,
    'comb_9_strides': 1,
    'comb_9_pad': 'same',

    # 5
    'comb_10_filters': 60,
    'comb_10_kernel': 2,
    'comb_10_strides': 2,
    'comb_10_pad': 'same',

    'comb_spatial_5': 0.4,

    'comb_11_filters': 60,
    'comb_11_kernel': 2,
    'comb_11_strides': 1,
    'comb_11_pad': 'same',

    'comb_12_filters': 60,
    'comb_12_kernel': 1,
    'comb_12_strides': 2,
    'comb_12_pad': 'same',

    # 6
    'comb_13_filters': 60,
    'comb_13_kernel': 2,
    'comb_13_strides': 2,
    'comb_13_pad': 'same',

    'comb_spatial_6': 0.4,

    'comb_14_filters': 60,
    'comb_14_kernel': 2,
    'comb_14_strides': 1,
    'comb_14_pad': 'same',

    'comb_15_filters': 60,
    'comb_15_kernel': 1,
    'comb_15_strides': 2,
    'comb_15_pad': 'same',

    # 7
    'comb_16_filters': 60,
    'comb_16_kernel': 2,
    'comb_16_strides': 2,
    'comb_16_pad': 'same',

    'comb_spatial_7': 0.4,

    'comb_17_filters': 60,
    'comb_17_kernel': 2,
    'comb_17_strides': 1,
    'comb_17_pad': 'same',

    'comb_18_filters': 60,
    'comb_18_kernel': 1,
    'comb_18_strides': 2,
    'comb_18_pad': 'same',

    # ---
    # Final Convs After Fluoro
    'comb_19_filters': 60,
    'comb_19_kernel': 2,
    'comb_19_strides': 1,
    'comb_19_pad': 'valid',

    # ---
    # Dense After Fluoro Convs
    'dense_comb_0_units': 50,
    'drop_1_comb': 0.3,
    'dense_comb_1_units': 50,

    # ---
    # Activation Function for Fluoro Vox Comb
    'flu_vox_act_fn': 'elu',

    # ---
    # Combine Fluoro and Vox
    'vox_flu_units_0': 60,
    'vox_flu_drop_1': 0.3,
    'vox_flu_units_1': 50,
    'vox_flu_drop_2': 0.3,
    'vox_flu_units_2': 30,
    'vox_flu_drop_3': 0.3,
    'vox_flu_units_3': 15,
    'vox_flu_units_4': 6,

    # ---
    # Cali Units
    'cali_0_units': 20,
    'drop_1_cali': 0.3,
    'cali_1_units': 20,
    'drop_2_cali': 0.3,
    'cali_2_units': 20,
    'cali_3_units': 6,

    # ---
    # Activation Function for Top Level Comb
    'top_level_act_fn': 'elu',
    'top_level_intra': None,

    # ---
    # Top Level Dense
    'top_drop_0': 0.2,
    'top_dense_0': 6,
    'top_dense_1': 6,

    'top_dense_2': 6,
    'top_drop_1': 0.2,
    'top_dense_3': 6,

    'top_dense_4': 6,
    'top_drop_2': 0.2,
    'top_dense_5': 6,

    'top_dense_6': 6,
    'top_drop_3': 0.2,
    'top_dense_7': 6,

    'top_dense_8': 6,
    'top_drop_4': 0.2,
    'top_dense_9': 6,

    'top_dense_10': 6,
    'top_drop_5': 0.2,
    'top_dense_11': 6,

    'top_dense_12': 6,


    # Main Output
    'main_output_units': 6,
    'main_output_act': 'linear',

    # General Housekeeping
    'v_conv_regularizer': tf.keras.regularizers.l1(1e-7),
    'conv_regularizer': tf.keras.regularizers.l1(1e-7),
    'dense_regularizer_1': tf.keras.regularizers.l1(1e-7),
    'dense_regularizer_2': tf.keras.regularizers.l1(1e-7),

    'activation_fn': 'elu',


    'kern_init': 'he_uniform',
    'model_opt': tf.keras.optimizers.Adam,
    'learning_rate': 0.001,
    'model_epochs': 40,
    'model_batchsize': 5,
    'model_loss': 'mae',
    'model_metric': 'mae'

}

# -----------------------------------------------------------------

channel_order = 'channels_last'
img_input_shape = (128, 128, 1)
vox_input_shape = (197, 162, 564, 1)
cali_input_shape = (6,)


# Input Layers
input_vox = tf.keras.Input(shape=vox_input_shape, name='input_vox', dtype='float32')
input_fluoro_1 = tf.keras.Input(shape=img_input_shape, name='input_fluoro_1', dtype='float32')
input_fluoro_2 = tf.keras.Input(shape=img_input_shape, name='input_fluoro_2', dtype='float32')
input_cali = tf.keras.Input(shape=cali_input_shape, name='input_cali', dtype='float32')

# -----------------------------------------------------------------
# VOXEL CONVS
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

bn_2 = tf.keras.layers.BatchNormalization()(v_conv_2)

# 1
v_conv_3 = tf.keras.layers.Conv3D(filters=params['v_conv_3_filters'], kernel_size=params['v_conv_3_kernel'], strides=(params['v_conv_3_strides_0'], params['v_conv_3_strides_1'], params['v_conv_3_strides_2']), padding=params['v_conv_3_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['v_conv_regularizer'])(bn_2)
bn_3 = tf.keras.layers.BatchNormalization()(v_conv_3)
v_spat_2 = tf.keras.layers.SpatialDropout3D(rate=params['v_spatial_drop_rate_2'])(bn_3)
v_conv_4 = tf.keras.layers.Conv3D(filters=params['v_conv_4_filters'], kernel_size=params['v_conv_4_kernel'], strides=(params['v_conv_4_strides_0'], params['v_conv_4_strides_1'], params['v_conv_4_strides_2']), padding=params['v_conv_4_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['v_intra_act_fn'])(v_spat_2)
bn_4 = tf.keras.layers.BatchNormalization()(v_conv_4)
v_add_0 = tf.keras.layers.Add()([bn_4, bn_2])
v_act_0 = tf.keras.layers.Activation(activation=params['v_res_act_fn'])(v_add_0)

# 2
v_conv_5 = tf.keras.layers.Conv3D(filters=params['v_conv_5_filters'], kernel_size=params['v_conv_5_kernel'], strides=(params['v_conv_5_strides_0'], params['v_conv_5_strides_1'], params['v_conv_5_strides_2']), padding=params['v_conv_5_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['v_conv_regularizer'])(v_act_0)
bn_5 = tf.keras.layers.BatchNormalization()(v_conv_5)
v_spat_3 = tf.keras.layers.SpatialDropout3D(rate=params['v_spatial_drop_rate_3'])(bn_5)
v_conv_6 = tf.keras.layers.Conv3D(filters=params['v_conv_6_filters'], kernel_size=params['v_conv_6_kernel'], strides=(params['v_conv_6_strides_0'], params['v_conv_6_strides_1'], params['v_conv_6_strides_2']), padding=params['v_conv_6_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['v_intra_act_fn'])(v_spat_3)
bn_6 = tf.keras.layers.BatchNormalization()(v_conv_6)
v_add_1 = tf.keras.layers.Add()([bn_6, v_act_0])
v_act_1 = tf.keras.layers.Activation(activation=params['v_res_act_fn'])(v_add_1)

# 3
v_conv_7 = tf.keras.layers.Conv3D(filters=params['v_conv_7_filters'], kernel_size=params['v_conv_7_kernel'], strides=(params['v_conv_7_strides_0'], params['v_conv_7_strides_1'], params['v_conv_7_strides_2']), padding=params['v_conv_7_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['v_conv_regularizer'])(v_act_1)
bn_7 = tf.keras.layers.BatchNormalization()(v_conv_7)
v_spat_4 = tf.keras.layers.SpatialDropout3D(rate=params['v_spatial_drop_rate_4'])(bn_7)
v_conv_8 = tf.keras.layers.Conv3D(filters=params['v_conv_8_filters'], kernel_size=params['v_conv_8_kernel'], strides=(params['v_conv_8_strides_0'], params['v_conv_8_strides_1'], params['v_conv_8_strides_2']), padding=params['v_conv_8_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['v_intra_act_fn'])(v_spat_4)
bn_8 = tf.keras.layers.BatchNormalization()(v_conv_8)
v_add_2 = tf.keras.layers.Add()([bn_8, v_act_1])
v_act_2 = tf.keras.layers.Activation(activation=params['v_res_act_fn'])(v_add_2)

# 4
v_conv_9 = tf.keras.layers.Conv3D(filters=params['v_conv_9_filters'], kernel_size=params['v_conv_9_kernel'], strides=(params['v_conv_9_strides_0'], params['v_conv_9_strides_1'], params['v_conv_9_strides_2']), padding=params['v_conv_9_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['v_conv_regularizer'])(v_act_2)
bn_9 = tf.keras.layers.BatchNormalization()(v_conv_9)
v_spat_5 = tf.keras.layers.SpatialDropout3D(rate=params['v_spatial_drop_rate_5'])(bn_9)
v_conv_10 = tf.keras.layers.Conv3D(filters=params['v_conv_10_filters'], kernel_size=params['v_conv_10_kernel'], strides=(params['v_conv_10_strides_0'], params['v_conv_10_strides_1'], params['v_conv_10_strides_2']), padding=params['v_conv_10_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['v_intra_act_fn'])(v_spat_5)
bn_10 = tf.keras.layers.BatchNormalization()(v_conv_10)
v_conv_11 = tf.keras.layers.Conv3D(filters=params['v_conv_11_filters'], kernel_size=params['v_conv_11_kernel'], strides=(params['v_conv_11_strides_0'], params['v_conv_11_strides_1'], params['v_conv_11_strides_2']), padding=params['v_conv_11_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['v_intra_act_fn'])(v_act_2)
bn_11 = tf.keras.layers.BatchNormalization()(v_conv_11)
v_add_3 = tf.keras.layers.Add()([bn_10, bn_11])
v_act_3 = tf.keras.layers.Activation(activation=params['v_res_act_fn'])(v_add_3)

# 5
v_conv_12 = tf.keras.layers.Conv3D(filters=params['v_conv_12_filters'], kernel_size=params['v_conv_12_kernel'], strides=(params['v_conv_12_strides_0'], params['v_conv_12_strides_1'], params['v_conv_12_strides_2']), padding=params['v_conv_12_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['v_conv_regularizer'])(v_act_3)
bn_12 = tf.keras.layers.BatchNormalization()(v_conv_12)
v_spat_6 = tf.keras.layers.SpatialDropout3D(rate=params['v_spatial_drop_rate_6'])(bn_12)
v_conv_13 = tf.keras.layers.Conv3D(filters=params['v_conv_13_filters'], kernel_size=params['v_conv_13_kernel'], strides=(params['v_conv_13_strides_0'], params['v_conv_13_strides_1'], params['v_conv_13_strides_2']), padding=params['v_conv_13_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['v_intra_act_fn'])(v_spat_6)
bn_13 = tf.keras.layers.BatchNormalization()(v_conv_13)
v_conv_14 = tf.keras.layers.Conv3D(filters=params['v_conv_14_filters'], kernel_size=params['v_conv_14_kernel'], strides=(params['v_conv_14_strides_0'], params['v_conv_14_strides_1'], params['v_conv_14_strides_2']), padding=params['v_conv_14_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['v_intra_act_fn'])(v_act_3)
bn_14 = tf.keras.layers.BatchNormalization()(v_conv_14)
v_add_4 = tf.keras.layers.Add()([bn_13, bn_14])
v_act_4 = tf.keras.layers.Activation(activation=params['v_res_act_fn'])(v_add_4)

# 6
v_conv_15 = tf.keras.layers.Conv3D(filters=params['v_conv_12_filters'], kernel_size=params['v_conv_12_kernel'], strides=(params['v_conv_12_strides_0'], params['v_conv_12_strides_1'], params['v_conv_12_strides_2']), padding=params['v_conv_12_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['v_conv_regularizer'])(v_act_4)
bn_15 = tf.keras.layers.BatchNormalization()(v_conv_15)
v_spat_7 = tf.keras.layers.SpatialDropout3D(rate=params['v_spatial_drop_rate_6'])(bn_15)
v_conv_16 = tf.keras.layers.Conv3D(filters=params['v_conv_13_filters'], kernel_size=params['v_conv_13_kernel'], strides=(params['v_conv_13_strides_0'], params['v_conv_13_strides_1'], params['v_conv_13_strides_2']), padding=params['v_conv_13_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['v_intra_act_fn'])(v_spat_7)
bn_16 = tf.keras.layers.BatchNormalization()(v_conv_16)
v_conv_17 = tf.keras.layers.Conv3D(filters=params['v_conv_14_filters'], kernel_size=params['v_conv_14_kernel'], strides=(params['v_conv_14_strides_0'], params['v_conv_14_strides_1'], params['v_conv_14_strides_2']), padding=params['v_conv_14_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['v_intra_act_fn'])(v_act_4)
bn_17 = tf.keras.layers.BatchNormalization()(v_conv_17)
v_add_5 = tf.keras.layers.Add()([bn_16, bn_17])
v_act_5 = tf.keras.layers.Activation(activation=params['v_res_act_fn'])(v_add_5)


# ---
# Final Conv Layers
bn_18 = tf.keras.layers.BatchNormalization()(v_act_5)
v_spat_8 = tf.keras.layers.SpatialDropout3D(rate=params['v_spatial_drop_rate_8'])(bn_18)
v_conv_18 = tf.keras.layers.Conv3D(filters=params['v_conv_18_filters'], kernel_size=params['v_conv_18_kernel'], strides=(params['v_conv_18_strides_0'], params['v_conv_18_strides_1'], params['v_conv_18_strides_2']), padding=params['v_conv_18_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['v_conv_regularizer'])(v_spat_8)

# ---
# Dense Layers
v_flatten_0 = tf.keras.layers.Flatten()(v_conv_18)

bn_19 = tf.keras.layers.BatchNormalization()(v_flatten_0)
dense_1_v_drop = tf.keras.layers.Dropout(params['drop_1_v_rate'])(bn_19)
dense_1_v = tf.keras.layers.Dense(units=params['dense_1_v_units'], activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['dense_regularizer_1'])(dense_1_v_drop)

bn_20 = tf.keras.layers.BatchNormalization()(dense_1_v)
dense_2_v_drop = tf.keras.layers.Dropout(params['drop_2_v_rate'])(bn_20)
dense_2_v = tf.keras.layers.Dense(units=params['dense_2_v_units'], activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['dense_regularizer_1'])(dense_2_v_drop)

bn_21_v = tf.keras.layers.BatchNormalization()(dense_2_v)


# -----------------------------------------------------------------
# FLUORO ANALYSIS 1
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

bn_2 = tf.keras.layers.BatchNormalization()(pool_0_1)

# 1
conv_2_1 = tf.keras.layers.Conv2D(filters=params['conv_2_filters'], kernel_size=params['conv_2_kernel'], strides=params['conv_2_strides'], padding=params['conv_2_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['conv_regularizer'])(bn_2)
bn_3 = tf.keras.layers.BatchNormalization()(conv_2_1)
spat_1_1 = tf.keras.layers.SpatialDropout2D(rate=params['spatial_drop_rate_1'])(bn_3)
conv_3_1 = tf.keras.layers.Conv2D(filters=params['conv_3_filters'], kernel_size=params['conv_3_kernel'], strides=params['conv_3_strides'], padding=params['conv_3_pad'], data_format=channel_order, activation=params['intra_act_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['conv_regularizer'])(spat_1_1)
bn_4 = tf.keras.layers.BatchNormalization()(conv_3_1)
add_0 = tf.keras.layers.Add()([bn_4, bn_2])
act_0 = tf.keras.layers.Activation(activation=params['res_act_fn'])(add_0)

# 2
conv_4_1 = tf.keras.layers.Conv2D(filters=params['conv_4_filters'], kernel_size=params['conv_4_kernel'], strides=params['conv_4_strides'], padding=params['conv_4_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['conv_regularizer'])(act_0)
bn_5 = tf.keras.layers.BatchNormalization()(conv_4_1)
spat_2_1 = tf.keras.layers.SpatialDropout2D(rate=params['spatial_drop_rate_1'])(bn_5)
conv_5_1 = tf.keras.layers.Conv2D(filters=params['conv_3_filters'], kernel_size=params['conv_3_kernel'], strides=params['conv_3_strides'], padding=params['conv_3_pad'], data_format=channel_order, activation=params['intra_act_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['conv_regularizer'])(spat_2_1)
bn_6 = tf.keras.layers.BatchNormalization()(conv_5_1)
add_1 = tf.keras.layers.Add()([act_0, bn_6])
act_1 = tf.keras.layers.Activation(activation=params['res_act_fn'])(add_1)

# 3
conv_6_1 = tf.keras.layers.Conv2D(filters=params['conv_6_filters'], kernel_size=params['conv_6_kernel'], strides=params['conv_6_strides'], padding=params['conv_6_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['conv_regularizer'])(act_1)
bn_7 = tf.keras.layers.BatchNormalization()(conv_6_1)
spat_3_1 = tf.keras.layers.SpatialDropout2D(rate=params['spatial_drop_rate_3'])(bn_7)
conv_7_1 = tf.keras.layers.Conv2D(filters=params['conv_7_filters'], kernel_size=params['conv_7_kernel'], strides=params['conv_7_strides'], padding=params['conv_7_pad'], data_format=channel_order, activation=params['intra_act_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['conv_regularizer'])(spat_3_1)
bn_8 = tf.keras.layers.BatchNormalization()(conv_7_1)
add_2 = tf.keras.layers.Add()([act_1, bn_8])
act_2 = tf.keras.layers.Activation(activation=params['res_act_fn'])(add_2)

# 4
conv_8_1 = tf.keras.layers.Conv2D(filters=params['conv_8_filters'], kernel_size=params['conv_8_kernel'], strides=params['conv_8_strides'], padding=params['conv_8_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['conv_regularizer'])(act_2)
bn_9 = tf.keras.layers.BatchNormalization()(conv_8_1)
spat_4_1 = tf.keras.layers.SpatialDropout2D(rate=params['spatial_drop_rate_4'])(bn_9)
conv_9_1 = tf.keras.layers.Conv2D(filters=params['conv_9_filters'], kernel_size=params['conv_9_kernel'], strides=params['conv_9_strides'], padding=params['conv_9_pad'], data_format=channel_order, activation=params['intra_act_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['conv_regularizer'])(spat_4_1)
bn_10 = tf.keras.layers.BatchNormalization()(conv_9_1)
add_3 = tf.keras.layers.Add()([act_2, bn_10])
act_3 = tf.keras.layers.Activation(activation=params['res_act_fn'])(add_3)

# 5
conv_10_1 = tf.keras.layers.Conv2D(filters=params['conv_10_filters'], kernel_size=params['conv_10_kernel'], strides=params['conv_10_strides'], padding=params['conv_10_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['conv_regularizer'])(act_3)
bn_11 = tf.keras.layers.BatchNormalization()(conv_10_1)
spat_5_1 = tf.keras.layers.SpatialDropout2D(rate=params['spatial_drop_rate_5'])(bn_11)
conv_11_1 = tf.keras.layers.Conv2D(filters=params['conv_11_filters'], kernel_size=params['conv_11_kernel'], strides=params['conv_11_strides'], padding=params['conv_11_pad'], data_format=channel_order, activation=params['intra_act_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['conv_regularizer'])(spat_5_1)
bn_12 = tf.keras.layers.BatchNormalization()(conv_11_1)
add_4 = tf.keras.layers.Add()([act_3, bn_12])
act_4 = tf.keras.layers.Activation(activation=params['res_act_fn'])(add_4)

# 6
conv_12_1 = tf.keras.layers.Conv2D(filters=params['conv_12_filters'], kernel_size=params['conv_12_kernel'], strides=params['conv_12_strides'], padding=params['conv_12_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['conv_regularizer'])(act_4)
bn_13 = tf.keras.layers.BatchNormalization()(conv_12_1)
spat_6_1 = tf.keras.layers.SpatialDropout2D(rate=params['spatial_drop_rate_6'])(bn_13)
conv_13_1 = tf.keras.layers.Conv2D(filters=params['conv_13_filters'], kernel_size=params['conv_13_kernel'], strides=params['conv_13_strides'], padding=params['conv_13_pad'], data_format=channel_order, activation=params['intra_act_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['conv_regularizer'])(spat_6_1)
bn_14 = tf.keras.layers.BatchNormalization()(conv_13_1)
add_5 = tf.keras.layers.Add()([act_4, bn_14])
act_5_1 = tf.keras.layers.Activation(activation=params['res_act_fn'])(add_5)



# -----------------------------------------------------------------
# FLUORO ANALYSIS 2
# -----------------------------------------------------------------

# ---
# Entry Fluoro Layers
per_image_stand_2 = tf.keras.layers.Lambda(lambda frame: tf.image.per_image_standardization(frame))(input_fluoro_2)

bn_0 = tf.keras.layers.BatchNormalization()(per_image_stand_2)
conv_0_1 = tf.keras.layers.Conv2D(filters=params['conv_0_filters'], kernel_size=params['conv_0_kernel'], strides=params['conv_0_strides'], padding=params['conv_0_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['conv_regularizer'])(bn_0)

bn_1 = tf.keras.layers.BatchNormalization()(conv_0_1)
spat_0_1 = tf.keras.layers.SpatialDropout2D(rate=params['spatial_drop_rate_0'])(bn_1)
conv_1_1 = tf.keras.layers.Conv2D(filters=params['conv_1_filters'], kernel_size=params['conv_1_kernel'], strides=params['conv_1_strides'], padding=params['conv_1_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['conv_regularizer'])(spat_0_1)

# ---
# Pool After Initial Layers
pool_0_1 = tf.keras.layers.AveragePooling2D(pool_size=params['pool_0_size'], padding=params['pool_0_pad'])(conv_1_1)

# ---
# Run of Residual Layers

bn_2 = tf.keras.layers.BatchNormalization()(pool_0_1)

# 1
conv_2_1 = tf.keras.layers.Conv2D(filters=params['conv_2_filters'], kernel_size=params['conv_2_kernel'], strides=params['conv_2_strides'], padding=params['conv_2_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['conv_regularizer'])(bn_2)
bn_3 = tf.keras.layers.BatchNormalization()(conv_2_1)
spat_1_1 = tf.keras.layers.SpatialDropout2D(rate=params['spatial_drop_rate_1'])(bn_3)
conv_3_1 = tf.keras.layers.Conv2D(filters=params['conv_3_filters'], kernel_size=params['conv_3_kernel'], strides=params['conv_3_strides'], padding=params['conv_3_pad'], data_format=channel_order, activation=params['intra_act_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['conv_regularizer'])(spat_1_1)
bn_4 = tf.keras.layers.BatchNormalization()(conv_3_1)
add_0 = tf.keras.layers.Add()([bn_4, bn_2])
act_0 = tf.keras.layers.Activation(activation=params['res_act_fn'])(add_0)

# 2
conv_4_1 = tf.keras.layers.Conv2D(filters=params['conv_4_filters'], kernel_size=params['conv_4_kernel'], strides=params['conv_4_strides'], padding=params['conv_4_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['conv_regularizer'])(act_0)
bn_5 = tf.keras.layers.BatchNormalization()(conv_4_1)
spat_2_1 = tf.keras.layers.SpatialDropout2D(rate=params['spatial_drop_rate_1'])(bn_5)
conv_5_1 = tf.keras.layers.Conv2D(filters=params['conv_3_filters'], kernel_size=params['conv_3_kernel'], strides=params['conv_3_strides'], padding=params['conv_3_pad'], data_format=channel_order, activation=params['intra_act_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['conv_regularizer'])(spat_2_1)
bn_6 = tf.keras.layers.BatchNormalization()(conv_5_1)
add_1 = tf.keras.layers.Add()([act_0, bn_6])
act_1 = tf.keras.layers.Activation(activation=params['res_act_fn'])(add_1)

# 3
conv_6_1 = tf.keras.layers.Conv2D(filters=params['conv_6_filters'], kernel_size=params['conv_6_kernel'], strides=params['conv_6_strides'], padding=params['conv_6_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['conv_regularizer'])(act_1)
bn_7 = tf.keras.layers.BatchNormalization()(conv_6_1)
spat_3_1 = tf.keras.layers.SpatialDropout2D(rate=params['spatial_drop_rate_3'])(bn_7)
conv_7_1 = tf.keras.layers.Conv2D(filters=params['conv_7_filters'], kernel_size=params['conv_7_kernel'], strides=params['conv_7_strides'], padding=params['conv_7_pad'], data_format=channel_order, activation=params['intra_act_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['conv_regularizer'])(spat_3_1)
bn_8 = tf.keras.layers.BatchNormalization()(conv_7_1)
add_2 = tf.keras.layers.Add()([act_1, bn_8])
act_2 = tf.keras.layers.Activation(activation=params['res_act_fn'])(add_2)

# 4
conv_8_1 = tf.keras.layers.Conv2D(filters=params['conv_8_filters'], kernel_size=params['conv_8_kernel'], strides=params['conv_8_strides'], padding=params['conv_8_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['conv_regularizer'])(act_2)
bn_9 = tf.keras.layers.BatchNormalization()(conv_8_1)
spat_4_1 = tf.keras.layers.SpatialDropout2D(rate=params['spatial_drop_rate_4'])(bn_9)
conv_9_1 = tf.keras.layers.Conv2D(filters=params['conv_9_filters'], kernel_size=params['conv_9_kernel'], strides=params['conv_9_strides'], padding=params['conv_9_pad'], data_format=channel_order, activation=params['intra_act_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['conv_regularizer'])(spat_4_1)
bn_10 = tf.keras.layers.BatchNormalization()(conv_9_1)
add_3 = tf.keras.layers.Add()([act_2, bn_10])
act_3 = tf.keras.layers.Activation(activation=params['res_act_fn'])(add_3)

# 5
conv_10_1 = tf.keras.layers.Conv2D(filters=params['conv_10_filters'], kernel_size=params['conv_10_kernel'], strides=params['conv_10_strides'], padding=params['conv_10_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['conv_regularizer'])(act_3)
bn_11 = tf.keras.layers.BatchNormalization()(conv_10_1)
spat_5_1 = tf.keras.layers.SpatialDropout2D(rate=params['spatial_drop_rate_5'])(bn_11)
conv_11_1 = tf.keras.layers.Conv2D(filters=params['conv_11_filters'], kernel_size=params['conv_11_kernel'], strides=params['conv_11_strides'], padding=params['conv_11_pad'], data_format=channel_order, activation=params['intra_act_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['conv_regularizer'])(spat_5_1)
bn_12 = tf.keras.layers.BatchNormalization()(conv_11_1)
add_4 = tf.keras.layers.Add()([act_3, bn_12])
act_4 = tf.keras.layers.Activation(activation=params['res_act_fn'])(add_4)

# 6
conv_12_1 = tf.keras.layers.Conv2D(filters=params['conv_12_filters'], kernel_size=params['conv_12_kernel'], strides=params['conv_12_strides'], padding=params['conv_12_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['conv_regularizer'])(act_4)
bn_13 = tf.keras.layers.BatchNormalization()(conv_12_1)
spat_6_1 = tf.keras.layers.SpatialDropout2D(rate=params['spatial_drop_rate_6'])(bn_13)
conv_13_1 = tf.keras.layers.Conv2D(filters=params['conv_13_filters'], kernel_size=params['conv_13_kernel'], strides=params['conv_13_strides'], padding=params['conv_13_pad'], data_format=channel_order, activation=params['intra_act_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['conv_regularizer'])(spat_6_1)
bn_14 = tf.keras.layers.BatchNormalization()(conv_13_1)
add_5 = tf.keras.layers.Add()([act_4, bn_14])
act_5_2 = tf.keras.layers.Activation(activation=params['res_act_fn'])(add_5)


# -----------------------------------------------------------------
# COMBINE FLUOROS
# -----------------------------------------------------------------

comb_fluoro_0 = tf.keras.layers.concatenate([act_5_1, act_5_2])


# -----------------------------------------------------------------
# RES NETS AFTER COMBINED FLUORO
# -----------------------------------------------------------------

# 0
comb_0 = tf.keras.layers.Conv2D(filters=params['comb_0_filters'], kernel_size=params['comb_0_kernel'], strides=params['comb_0_strides'], padding=params['comb_0_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['conv_regularizer'])(comb_fluoro_0)
bn_0 = tf.keras.layers.BatchNormalization()(comb_0)
spat_0 = tf.keras.layers.SpatialDropout2D(rate=params['comb_spatial_0'])(bn_0)
comb_1 = tf.keras.layers.Conv2D(filters=params['comb_1_filters'], kernel_size=params['comb_1_kernel'], strides=params['comb_1_strides'], padding=params['comb_1_pad'], data_format=channel_order, activation=params['c_intra_act_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['conv_regularizer'])(spat_0)
bn_1 = tf.keras.layers.BatchNormalization()(comb_1)
add_0 = tf.keras.layers.Add()([comb_fluoro_0, bn_1])
act_0 = tf.keras.layers.Activation(activation=params['c_res_act_fn'])(add_0)

# 1
comb_2 = tf.keras.layers.Conv2D(filters=params['comb_2_filters'], kernel_size=params['comb_2_kernel'], strides=params['comb_2_strides'], padding=params['comb_2_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['conv_regularizer'])(act_0)
bn_2 = tf.keras.layers.BatchNormalization()(comb_2)
spat_1 = tf.keras.layers.SpatialDropout2D(rate=params['comb_spatial_1'])(bn_2)
comb_3 = tf.keras.layers.Conv2D(filters=params['comb_3_filters'], kernel_size=params['comb_3_kernel'], strides=params['comb_3_strides'], padding=params['comb_3_pad'], data_format=channel_order, activation=params['c_intra_act_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['conv_regularizer'])(spat_1)
bn_3 = tf.keras.layers.BatchNormalization()(comb_3)
add_1 = tf.keras.layers.Add()([act_0, bn_3])
act_1 = tf.keras.layers.Activation(activation=params['c_res_act_fn'])(add_1)

# 2
comb_4 = tf.keras.layers.Conv2D(filters=params['comb_4_filters'], kernel_size=params['comb_4_kernel'], strides=params['comb_4_strides'], padding=params['comb_4_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['conv_regularizer'])(act_1)
bn_4 = tf.keras.layers.BatchNormalization()(comb_4)
spat_2 = tf.keras.layers.SpatialDropout2D(rate=params['comb_spatial_2'])(bn_4)
comb_5 = tf.keras.layers.Conv2D(filters=params['comb_5_filters'], kernel_size=params['comb_5_kernel'], strides=params['comb_5_strides'], padding=params['comb_5_pad'], data_format=channel_order, activation=params['c_intra_act_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['conv_regularizer'])(spat_2)
bn_5 = tf.keras.layers.BatchNormalization()(comb_5)
add_2 = tf.keras.layers.Add()([act_1, bn_5])
act_2 = tf.keras.layers.Activation(activation=params['c_res_act_fn'])(add_2)

# 3
comb_6 = tf.keras.layers.Conv2D(filters=params['comb_6_filters'], kernel_size=params['comb_6_kernel'], strides=params['comb_6_strides'], padding=params['comb_6_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['conv_regularizer'])(act_2)
bn_6 = tf.keras.layers.BatchNormalization()(comb_6)
spat_3 = tf.keras.layers.SpatialDropout2D(rate=params['comb_spatial_3'])(bn_6)
comb_7 = tf.keras.layers.Conv2D(filters=params['comb_7_filters'], kernel_size=params['comb_7_kernel'], strides=params['comb_7_strides'], padding=params['comb_7_pad'], data_format=channel_order, activation=params['c_intra_act_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['conv_regularizer'])(spat_3)
bn_7 = tf.keras.layers.BatchNormalization()(comb_7)
add_3 = tf.keras.layers.Add()([act_2, bn_7])
act_3 = tf.keras.layers.Activation(activation=params['c_res_act_fn'])(add_3)

# 4
comb_8 = tf.keras.layers.Conv2D(filters=params['comb_8_filters'], kernel_size=params['comb_8_kernel'], strides=params['comb_8_strides'], padding=params['comb_8_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['conv_regularizer'])(act_3)
bn_8 = tf.keras.layers.BatchNormalization()(comb_8)
spat_4 = tf.keras.layers.SpatialDropout2D(rate=params['comb_spatial_4'])(bn_8)
comb_9 = tf.keras.layers.Conv2D(filters=params['comb_9_filters'], kernel_size=params['comb_9_kernel'], strides=params['comb_9_strides'], padding=params['comb_9_pad'], data_format=channel_order, activation=params['c_intra_act_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['conv_regularizer'])(spat_4)
bn_9 = tf.keras.layers.BatchNormalization()(comb_9)
add_4 = tf.keras.layers.Add()([act_3, bn_9])
act_4 = tf.keras.layers.Activation(activation=params['c_res_act_fn'])(add_4)

# 5
comb_10 = tf.keras.layers.Conv2D(filters=params['comb_10_filters'], kernel_size=params['comb_10_kernel'], strides=params['comb_10_strides'], padding=params['comb_10_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['conv_regularizer'])(act_4)
bn_10 = tf.keras.layers.BatchNormalization()(comb_10)
spat_5 = tf.keras.layers.SpatialDropout2D(rate=params['comb_spatial_5'])(bn_10)
comb_11 = tf.keras.layers.Conv2D(filters=params['comb_11_filters'], kernel_size=params['comb_11_kernel'], strides=params['comb_11_strides'], padding=params['comb_11_pad'], data_format=channel_order, activation=params['c_intra_act_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['conv_regularizer'])(spat_5)
bn_11 = tf.keras.layers.BatchNormalization()(comb_11)
comb_12 = tf.keras.layers.Conv2D(filters=params['comb_12_filters'], kernel_size=params['comb_12_kernel'], strides=params['comb_12_strides'], padding=params['comb_12_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['conv_regularizer'])(act_4)
bn_12 = tf.keras.layers.BatchNormalization()(comb_12)
add_5 = tf.keras.layers.Add()([bn_11, bn_12])
act_5 = tf.keras.layers.Activation(activation=params['c_res_act_fn'])(add_5)

# 6
comb_13 = tf.keras.layers.Conv2D(filters=params['comb_13_filters'], kernel_size=params['comb_13_kernel'], strides=params['comb_13_strides'], padding=params['comb_13_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['conv_regularizer'])(act_5)
bn_13 = tf.keras.layers.BatchNormalization()(comb_13)
spat_6 = tf.keras.layers.SpatialDropout2D(rate=params['comb_spatial_6'])(bn_13)
comb_14 = tf.keras.layers.Conv2D(filters=params['comb_14_filters'], kernel_size=params['comb_14_kernel'], strides=params['comb_14_strides'], padding=params['comb_14_pad'], data_format=channel_order, activation=params['c_intra_act_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['conv_regularizer'])(spat_6)
bn_14 = tf.keras.layers.BatchNormalization()(comb_14)
comb_15 = tf.keras.layers.Conv2D(filters=params['comb_15_filters'], kernel_size=params['comb_15_kernel'], strides=params['comb_15_strides'], padding=params['comb_15_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['conv_regularizer'])(act_5)
bn_15 = tf.keras.layers.BatchNormalization()(comb_15)
add_6 = tf.keras.layers.Add()([bn_14, bn_15])
act_6 = tf.keras.layers.Activation(activation=params['c_res_act_fn'])(add_6)

# 7
comb_16 = tf.keras.layers.Conv2D(filters=params['comb_16_filters'], kernel_size=params['comb_16_kernel'], strides=params['comb_16_strides'], padding=params['comb_16_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['conv_regularizer'])(act_6)
bn_16 = tf.keras.layers.BatchNormalization()(comb_16)
spat_7 = tf.keras.layers.SpatialDropout2D(rate=params['comb_spatial_7'])(bn_16)
comb_17 = tf.keras.layers.Conv2D(filters=params['comb_17_filters'], kernel_size=params['comb_17_kernel'], strides=params['comb_17_strides'], padding=params['comb_17_pad'], data_format=channel_order, activation=params['c_intra_act_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['conv_regularizer'])(spat_7)
bn_17 = tf.keras.layers.BatchNormalization()(comb_17)
comb_18 = tf.keras.layers.Conv2D(filters=params['comb_18_filters'], kernel_size=params['comb_18_kernel'], strides=params['comb_18_strides'], padding=params['comb_18_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['conv_regularizer'])(act_6)
bn_18 = tf.keras.layers.BatchNormalization()(comb_18)
add_7 = tf.keras.layers.Add()([bn_17, bn_18])
act_7 = tf.keras.layers.Activation(activation=params['c_res_act_fn'])(add_7)


# ---
# Conv After End of Res Net
comb_19 = tf.keras.layers.Conv2D(filters=params['comb_19_filters'], kernel_size=params['comb_19_kernel'], strides=params['comb_19_strides'], padding=params['comb_19_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['conv_regularizer'])(act_7)

# ---
# Dense At End of Convs
comb_flatten_1 = tf.keras.layers.Flatten()(comb_19)

bn_19 = tf.keras.layers.BatchNormalization()(comb_flatten_1)
dense_0_comb = tf.keras.layers.Dense(units=params['dense_comb_0_units'], activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['dense_regularizer_1'])(bn_19)

bn_20 = tf.keras.layers.BatchNormalization()(dense_0_comb)
dense_comb_1 = tf.keras.layers.Dropout(params['drop_1_comb'])(bn_20)
dense_1_comb = tf.keras.layers.Dense(units=params['dense_comb_1_units'], activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['dense_regularizer_1'])(dense_comb_1)

bn_21_f = tf.keras.layers.BatchNormalization()(dense_1_comb)


# -----------------------------------------------------------------
# COMBINE FLUORO NETS AND VOXEL NETS
# -----------------------------------------------------------------

fluoro_vox_comb = tf.keras.layers.Add()([bn_21_f, bn_21_v])
fluoro_vox_act = tf.keras.layers.Activation(activation=params['flu_vox_act_fn'])(fluoro_vox_comb)

# -----------------------------------------------------------------
# DENSE AFTER FLUORO AND VOXEL
# -----------------------------------------------------------------

bn_0 = tf.keras.layers.BatchNormalization()(fluoro_vox_act)
vox_flu_0 = tf.keras.layers.Dense(units=params['vox_flu_units_0'], activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['dense_regularizer_1'])(bn_0)

bn_1 = tf.keras.layers.BatchNormalization()(vox_flu_0)
vox_flu_drop_1 = tf.keras.layers.Dropout(params['vox_flu_drop_1'])(bn_1)
vox_flu_1 = tf.keras.layers.Dense(units=params['vox_flu_units_1'], activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['dense_regularizer_1'])(vox_flu_drop_1)

bn_2 = tf.keras.layers.BatchNormalization()(vox_flu_1)
vox_flu_drop_2 = tf.keras.layers.Dropout(params['vox_flu_drop_2'])(bn_2)
vox_flu_2 = tf.keras.layers.Dense(units=params['vox_flu_units_2'], activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['dense_regularizer_1'])(vox_flu_drop_2)

bn_3 = tf.keras.layers.BatchNormalization()(vox_flu_2)
vox_flu_drop_3 = tf.keras.layers.Dropout(params['vox_flu_drop_3'])(bn_3)
vox_flu_3 = tf.keras.layers.Dense(units=params['vox_flu_units_3'], activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['dense_regularizer_1'])(vox_flu_drop_3)

bn_4 = tf.keras.layers.BatchNormalization()(vox_flu_3)
vox_flu_4 = tf.keras.layers.Dense(units=params['vox_flu_units_4'], activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['dense_regularizer_1'])(bn_4)

bn_5_comb = tf.keras.layers.BatchNormalization()(vox_flu_4)

# -----------------------------------------------------------------
# CALIBRATION DENSE
# -----------------------------------------------------------------

bn_0 = tf.keras.layers.BatchNormalization()(input_cali)
cali_0 = tf.keras.layers.Dense(units=params['cali_0_units'], activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['dense_regularizer_1'])(bn_0)

bn_1 = tf.keras.layers.BatchNormalization()(cali_0)
drop_1_cali = tf.keras.layers.Dropout(params['drop_1_cali'])(bn_1)
cali_1 = tf.keras.layers.Dense(units=params['cali_1_units'], activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['dense_regularizer_1'])(drop_1_cali)

bn_2 = tf.keras.layers.BatchNormalization()(cali_1)
drop_2_cali = tf.keras.layers.Dropout(params['drop_2_cali'])(bn_2)
cali_2 = tf.keras.layers.Dense(units=params['cali_2_units'], activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['dense_regularizer_1'])(drop_2_cali)

bn_3 = tf.keras.layers.BatchNormalization()(cali_2)
cali_3 = tf.keras.layers.Dense(units=params['cali_3_units'], activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['dense_regularizer_1'])(bn_3)

bn_4_c = tf.keras.layers.BatchNormalization()(cali_3)


# -----------------------------------------------------------------
# COMBINE CALI AND VOX/FLUORO
# -----------------------------------------------------------------

top_level_comb = tf.keras.layers.Add()([bn_4_c, bn_5_comb])
top_level_act = tf.keras.layers.Activation(activation=params['top_level_act_fn'])(top_level_comb)


# -----------------------------------------------------------------
# TOP LEVEL DENSE TO OUTPUT
# -----------------------------------------------------------------

top_dense_0 = tf.keras.layers.Dense(units=params['top_dense_0'], activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['dense_regularizer_2'])(top_level_act)
bn_0 = tf.keras.layers.BatchNormalization()(top_dense_0)
top_drop_0 = tf.keras.layers.Dropout(params['top_drop_0'])(bn_0)
top_dense_1 = tf.keras.layers.Dense(units=params['top_dense_1'], activation=params['top_level_intra'], kernel_initializer=params['kern_init'], activity_regularizer=params['dense_regularizer_2'])(top_drop_0)
bn_1 = tf.keras.layers.BatchNormalization()(top_dense_1)
add_0 = tf.keras.layers.Add()([bn_1, bn_4_c])
act_0 = tf.keras.layers.Activation(activation=params['top_level_act_fn'])(add_0)


top_dense_2 = tf.keras.layers.Dense(units=params['top_dense_2'], activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['dense_regularizer_2'])(act_0)
bn_2 = tf.keras.layers.BatchNormalization()(top_dense_2)
top_drop_1 = tf.keras.layers.Dropout(params['top_drop_1'])(bn_2)
top_dense_3 = tf.keras.layers.Dense(units=params['top_dense_3'], activation=params['top_level_intra'], kernel_initializer=params['kern_init'], activity_regularizer=params['dense_regularizer_2'])(top_drop_1)
bn_3 = tf.keras.layers.BatchNormalization()(top_dense_3)
add_1 = tf.keras.layers.Add()([bn_3, act_0])
act_1 = tf.keras.layers.Activation(activation=params['top_level_act_fn'])(add_1)


top_dense_4 = tf.keras.layers.Dense(units=params['top_dense_4'], activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['dense_regularizer_2'])(act_1)
bn_4 = tf.keras.layers.BatchNormalization()(top_dense_4)
top_drop_2 = tf.keras.layers.Dropout(params['top_drop_2'])(bn_4)
top_dense_5 = tf.keras.layers.Dense(units=params['top_dense_5'], activation=params['top_level_intra'], kernel_initializer=params['kern_init'], activity_regularizer=params['dense_regularizer_2'])(top_drop_2)
bn_5 = tf.keras.layers.BatchNormalization()(top_dense_5)
add_2 = tf.keras.layers.Add()([bn_5, act_1])
act_2 = tf.keras.layers.Activation(activation=params['top_level_act_fn'])(add_2)

top_dense_6 = tf.keras.layers.Dense(units=params['top_dense_6'], activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['dense_regularizer_2'])(act_2)
bn_6 = tf.keras.layers.BatchNormalization()(top_dense_6)
top_drop_3 = tf.keras.layers.Dropout(params['top_drop_3'])(bn_6)
top_dense_7 = tf.keras.layers.Dense(units=params['top_dense_7'], activation=params['top_level_intra'], kernel_initializer=params['kern_init'], activity_regularizer=params['dense_regularizer_2'])(top_drop_3)
bn_7 = tf.keras.layers.BatchNormalization()(top_dense_7)
add_3 = tf.keras.layers.Add()([bn_7, act_2])
act_3 = tf.keras.layers.Activation(activation=params['top_level_act_fn'])(add_3)

top_dense_8 = tf.keras.layers.Dense(units=params['top_dense_8'], activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['dense_regularizer_2'])(act_3)
bn_8 = tf.keras.layers.BatchNormalization()(top_dense_8)
top_drop_4 = tf.keras.layers.Dropout(params['top_drop_4'])(bn_8)
top_dense_9 = tf.keras.layers.Dense(units=params['top_dense_9'], activation=params['top_level_intra'], kernel_initializer=params['kern_init'], activity_regularizer=params['dense_regularizer_2'])(top_drop_4)
bn_9 = tf.keras.layers.BatchNormalization()(top_dense_9)
add_4 = tf.keras.layers.Add()([bn_9, act_3])
act_4 = tf.keras.layers.Activation(activation=params['top_level_act_fn'])(add_4)

top_dense_10 = tf.keras.layers.Dense(units=params['top_dense_10'], activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['dense_regularizer_2'])(act_4)
bn_10 = tf.keras.layers.BatchNormalization()(top_dense_10)
top_dense_11 = tf.keras.layers.Dense(units=params['top_dense_11'], activation=params['top_level_intra'], kernel_initializer=params['kern_init'], activity_regularizer=params['dense_regularizer_2'])(bn_10)
bn_11 = tf.keras.layers.BatchNormalization()(top_dense_11)
add_5 = tf.keras.layers.Add()([bn_11, act_4])
act_5 = tf.keras.layers.Activation(activation=params['top_level_act_fn'])(add_5)



top_dense_12 = tf.keras.layers.Dense(units=params['top_dense_4'], activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=None)(act_5)



# -----------------------------------------------------------------


# Main Output
main_output = tf.keras.layers.Dense(units=params['main_output_units'], activation=params['main_output_act'], kernel_initializer=params['kern_init'], name='main_output')(top_dense_12)

# -----------------------------------------------------------------

# Model Housekeeping
model = tf.keras.Model(inputs=[input_vox, input_fluoro_1, input_fluoro_2, input_cali], outputs=main_output)

model.compile(optimizer=params['model_opt'](lr=params['learning_rate']), loss=params['model_loss'], metrics=[params['model_metric']])
tf.keras.utils.plot_model(model, os.path.abspath(os.path.join(save_dir, expr_name + '_' + expr_no + '.png')), show_shapes=True)

model.summary()

# -----------------------------------------------------------------

vox_file = h5py.File(os.path.expanduser('~/fluoro/data/compilation/voxels_mark_origin_comp.h5py'), 'r')
vox_init = vox_file['vox_dset']

image_file = h5py.File(os.path.expanduser('~/fluoro/data/compilation/images.h5py'), 'r')
image_init = image_file['image_dset']

label_file = h5py.File(os.path.expanduser('~/fluoro/data/compilation/labels.h5py'), 'r')
label_init = label_file['labels_dset']

cali_file = h5py.File(os.path.expanduser('~/fluoro/data/compilation/calibration.h5py'), 'r')
cali_init = cali_file['cali_len3_rot']


def split_train_test(shape, num_of_samples=None, ratio=0.2):

    if num_of_samples is None:
        shuffled_indices = np.random.choice(shape, size=shape, replace=False)
    else:
        shuffled_indices = np.random.choice(shape, size=num_of_samples, replace=False)


    test_set_size = int(len(shuffled_indices) * 0.2)
    test_indx = shuffled_indices[:test_set_size]
    train_indx = shuffled_indices[test_set_size:]

    return test_indx, train_indx


num_of_samples = None


test_indxs, train_sup_indxs = split_train_test(len(label_init), num_of_samples=num_of_samples)
val_indxs, train_indxs = split_train_test(len(train_sup_indxs))

val_indxs = train_sup_indxs[val_indxs]
train_indxs = train_sup_indxs[train_indxs]

test_indxs = sorted(list(test_indxs))
val_indxs = sorted(list(val_indxs))
train_indxs = sorted(list(train_indxs))


hist_file = open(os.path.join(save_dir, expr_name + '_hist_objects_' + expr_no + '.pkl'), 'wb')

var_dict = {}

var_dict['test_indxs'] = test_indxs
var_dict['val_indxs'] = val_indxs
var_dict['train_indxs'] = train_indxs

vox_mat_train = vox_init[:]
vox_mat_val = vox_mat_train[val_indxs]
vox_mat_train = vox_mat_train[train_indxs]
vox_file.close()

image_mat_train = image_init[:]
image_mat_val = image_mat_train[val_indxs]
image_mat_train = image_mat_train[train_indxs]
image_file.close()

cali_mat_train = cali_init[:]
cali_mat_val = cali_mat_train[val_indxs]
cali_mat_train = cali_mat_train[train_indxs]
cali_file.close()

label_mat_train = label_init[:]
label_mat_val = label_mat_train[val_indxs]
label_mat_train = label_mat_train[train_indxs]
label_file.close()

# -----------------------------------------------------------------


print('\n\ncompletely loaded...\n\n')


result = model.fit(x={'input_vox': np.expand_dims(vox_mat_train, axis=-1), 'input_fluoro_1': np.expand_dims(image_mat_train[:, 0, :, :], axis=-1), 'input_fluoro_2': np.expand_dims(image_mat_train[:, 1, :, :], axis=-1), 'input_cali': cali_mat_train}, y=label_mat_train, validation_data=([np.expand_dims(vox_mat_val, axis=-1), np.expand_dims(image_mat_val[:, 0, :, :], axis=-1), np.expand_dims(image_mat_val[:, 1, :, :], axis=-1), cali_mat_val], label_mat_val), epochs=params['model_epochs'], batch_size=params['model_batchsize'], shuffle=True, verbose=2)

model.save(os.path.abspath(os.path.join(save_dir, expr_name + '_' + expr_no + '.h5')))


var_dict['result'] = result.history
pickle.dump(var_dict, hist_file)
hist_file.close()





