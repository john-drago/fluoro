import numpy as np
import h5py
import tensorflow as tf
import keras
import os
import json
import csv
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform



expr_no = '1'
save_dir = os.path.abspath(os.path.join(os.path.expanduser('~/fluoro/code/hyperparameter/vox_fluoro'), 'vox_fluoro_res_hyperas'))
os.makedirs(save_dir, exist_ok=True)


def data_comp(first_indx=None, last_indx=None, num_of_samples=5):

    vox_file = h5py.File(os.path.expanduser('~/fluoro/data/compilation/voxels_pad.h5py'), 'r')
    vox_init = vox_file['vox_dset']

    image_file = h5py.File(os.path.expanduser('~/fluoro/data/compilation/images.h5py'), 'r')
    image_init = image_file['image_dset']

    label_file = h5py.File(os.path.expanduser('~/fluoro/data/compilation/labels.h5py'), 'r')
    label_init = label_file['labels_dset']

    cali_file = h5py.File(os.path.expanduser('~/fluoro/data/compilation/calibration.h5py'), 'r')
    cali_init = cali_file['cali_len3_rot']


    def split_train_test(shape, num_of_samples=5, ratio=0.2):

        if num_of_samples is None:
            shuffled_indices = np.random.choice(shape, size=shape, replace=False)
        else:
            shuffled_indices = np.random.choice(shape, size=num_of_samples, replace=False)


        test_set_size = int(len(shuffled_indices) * 0.2)
        test_indx = shuffled_indices[:test_set_size]
        train_indx = shuffled_indices[test_set_size:]

        return test_indx, train_indx


    test_indxs, train_indxs = split_train_test(len(label_init), num_of_samples=5)

    test_indxs = sorted(list(test_indxs))
    train_indxs = sorted(list(train_indxs))

    vox_mat_train = vox_init[:]
    vox_mat_train = vox_mat_train[train_indxs]
    vox_file.close()

    image_mat_train = image_init[:]
    image_mat_train = image_mat_train[train_indxs]
    image_file.close()

    cali_mat_train = cali_init[:]
    cali_mat_train = cali_mat_train[train_indxs]
    cali_file.close()

    label_mat_train = label_init[:]
    label_mat_train = label_mat_train[train_indxs]
    label_file.close()

    return vox_mat_train, image_mat_train, cali_mat_train, label_mat_train


def fluoro_model(vox_mat_train, image_mat_train, cali_mat_train, label_mat_train):


    v_intra_act_fn = None
    v_res_act_fn = 'elu'

    v_conv_0_filters = 30
    v_conv_0_kernel = 9
    v_conv_0_strides_0 = 2
    v_conv_0_strides_1 = 2
    v_conv_0_strides_2 = 2
    v_conv_0_pad = 'same'


    v_spatial_drop_rate_0 = 0.3

    v_conv_1_filters = 30
    v_conv_1_kernel = 5
    v_conv_1_strides_0 = 2
    v_conv_1_strides_1 = 2
    v_conv_1_strides_2 = 3
    v_conv_1_pad = 'same'


    v_pool_0_size = 2
    v_pool_0_pad = 'same'


    v_conv_2_filters = 30
    v_conv_2_kernel = 5
    v_conv_2_strides_0 = 2
    v_conv_2_strides_1 = 2
    v_conv_2_strides_2 = 2
    v_conv_2_pad = 'same'


    v_conv_5_filters = 30
    v_conv_5_kernel = 3
    v_conv_5_strides_0 = 1
    v_conv_5_strides_1 = 1
    v_conv_5_strides_2 = 1
    v_conv_5_pad = 'same'

    v_spatial_drop_rate_3 = 0.3

    v_conv_6_filters = 30
    v_conv_6_kernel = 3
    v_conv_6_strides_0 = 1
    v_conv_6_strides_1 = 1
    v_conv_6_strides_2 = 1
    v_conv_6_pad = 'same'


    v_conv_9_filters = 40
    v_conv_9_kernel = 3
    v_conv_9_strides_0 = 2
    v_conv_9_strides_1 = 2
    v_conv_9_strides_2 = 2
    v_conv_9_pad = 'same'

    v_spatial_drop_rate_5 = 0.3

    v_conv_10_filters = 40
    v_conv_10_kernel = 3
    v_conv_10_strides_0 = 1
    v_conv_10_strides_1 = 1
    v_conv_10_strides_2 = 1
    v_conv_10_pad = 'same'

    v_conv_11_filters = 40
    v_conv_11_kernel = 3
    v_conv_11_strides_0 = 2
    v_conv_11_strides_1 = 2
    v_conv_11_strides_2 = 2
    v_conv_11_pad = 'same'


    v_spatial_drop_rate_8 = 0.5

    v_conv_18_filters = 50
    v_conv_18_kernel = 2
    v_conv_18_strides_0 = 1
    v_conv_18_strides_1 = 1
    v_conv_18_strides_2 = 1
    v_conv_18_pad = 'valid'

    dense_1_v_units = 75
    dense_2_v_units = 50


    intra_act_fn = None
    res_act_fn = 'elu'


    conv_0_filters = 30
    conv_0_kernel = 5
    conv_0_strides = 2
    conv_0_pad = 'same'

    spatial_drop_rate_0 = 0.3

    conv_1_filters = 30
    conv_1_kernel = 5
    conv_1_strides = 2
    conv_1_pad = 'same'


    pool_0_size = 2
    pool_0_pad = 'same'


    conv_2_filters = 30
    conv_2_kernel = 3
    conv_2_strides = 1
    conv_2_pad = 'same'

    spatial_drop_rate_1 = 0.3

    conv_3_filters = 30
    conv_3_kernel = 3
    conv_3_strides = 1
    conv_3_pad = 'same'


    c_intra_act_fn = None
    c_res_act_fn = 'elu'

    comb_0_filters = 60
    comb_0_kernel = 3
    comb_0_strides = 1
    comb_0_pad = 'same'

    comb_spatial_0 = 0.3

    comb_1_filters = 60
    comb_1_kernel = 3
    comb_1_strides = 1
    comb_1_pad = 'same'


    comb_10_filters = 60
    comb_10_kernel = 2
    comb_10_strides = 2
    comb_10_pad = 'same'

    comb_spatial_5 = 0.3

    comb_11_filters = 60
    comb_11_kernel = 2
    comb_11_strides = 1
    comb_11_pad = 'same'

    comb_12_filters = 60
    comb_12_kernel = 1
    comb_12_strides = 2
    comb_12_pad = 'same'


    comb_19_filters = 60
    comb_19_kernel = 2
    comb_19_strides = 1
    comb_19_pad = 'valid'


    dense_comb_0_units = 50
    dense_comb_1_units = 50


    flu_vox_act_fn = 'elu'


    vox_flu_units_0 = 60
    vox_flu_units_1 = 50
    vox_flu_units_2 = 30
    vox_flu_units_3 = 15
    vox_flu_units_4 = 6


    cali_0_units = 20
    cali_1_units = 20
    cali_2_units = 20
    cali_3_units = 6


    top_level_act_fn = 'elu'
    top_level_intra = None


    top_dense_0 = 6
    top_dense_1 = 6
    top_dense_2 = 6
    top_dense_3 = 6
    top_dense_4 = 6
    top_dense_5 = 6
    top_dense_6 = 6


    main_output_units = 6
    main_output_act = 'linear'

    v_conv_regularizer = None
    conv_regularizer = None
    dense_regularizer_1 = None
    dense_regularizer_2 = None

    activation_fn = 'elu'


    kern_init = 'glorot_uniform'
    model_opt = 'adam'
    learning_rate = 0.001
    model_epochs = {{choice([1, 2])}}
    model_batchsize = 3
    model_loss = 'mse'
    model_metric = 'mse'


    channel_order = 'channels_last'
    img_input_shape = (128, 128, 1)
    vox_input_shape = (199, 164, 566, 1)
    cali_input_shape = (6,)


    input_vox = tf.keras.Input(shape=vox_input_shape, name='input_vox', dtype='float32')
    input_fluoro_1 = tf.keras.Input(shape=img_input_shape, name='input_fluoro_1', dtype='float32')
    input_fluoro_2 = tf.keras.Input(shape=img_input_shape, name='input_fluoro_2', dtype='float32')
    input_cali = tf.keras.Input(shape=cali_input_shape, name='input_cali', dtype='float32')


    v_conv_0 = tf.keras.layers.Conv3D(filters=v_conv_0_filters, kernel_size=v_conv_0_kernel, strides=(v_conv_0_strides_0, v_conv_0_strides_1, v_conv_0_strides_2), padding=v_conv_0_pad, data_format=channel_order, activation=activation_fn, kernel_initializer=kern_init, activity_regularizer=v_conv_regularizer)(input_vox)
    bn_0 = tf.keras.layers.BatchNormalization()(v_conv_0)


    v_spat_0 = tf.keras.layers.SpatialDropout3D(rate=v_spatial_drop_rate_0)(bn_0)
    v_conv_1 = tf.keras.layers.Conv3D(filters=v_conv_1_filters, kernel_size=v_conv_1_kernel, strides=(v_conv_1_strides_0, v_conv_1_strides_1, v_conv_1_strides_2), padding=v_conv_1_pad, data_format=channel_order, activation=activation_fn, kernel_initializer=kern_init, activity_regularizer=v_conv_regularizer)(v_spat_0)


    v_pool_0 = tf.keras.layers.MaxPooling3D(pool_size=v_pool_0_size, padding=v_pool_0_pad, data_format=channel_order)(v_conv_1)


    bn_1 = tf.keras.layers.BatchNormalization()(v_pool_0)
    v_conv_2 = tf.keras.layers.Conv3D(filters=v_conv_2_filters, kernel_size=v_conv_2_kernel, strides=(v_conv_2_strides_0, v_conv_2_strides_1, v_conv_2_strides_2), padding=v_conv_2_pad, data_format=channel_order, activation=activation_fn, kernel_initializer=kern_init, activity_regularizer=v_conv_regularizer)(bn_1)


    bn_2 = tf.keras.layers.BatchNormalization()(v_conv_2)

    v_conv_5 = tf.keras.layers.Conv3D(filters=v_conv_5_filters, kernel_size=v_conv_5_kernel, strides=(v_conv_5_strides_0, v_conv_5_strides_1, v_conv_5_strides_2), padding=v_conv_5_pad, data_format=channel_order, activation=activation_fn, kernel_initializer=kern_init, activity_regularizer=v_conv_regularizer)(bn_2)
    bn_5 = tf.keras.layers.BatchNormalization()(v_conv_5)
    v_spat_3 = tf.keras.layers.SpatialDropout3D(rate=v_spatial_drop_rate_3)(bn_5)
    v_conv_6 = tf.keras.layers.Conv3D(filters=v_conv_6_filters, kernel_size=v_conv_6_kernel, strides=(v_conv_6_strides_0, v_conv_6_strides_1, v_conv_6_strides_2), padding=v_conv_6_pad, data_format=channel_order, activation=activation_fn, kernel_initializer=kern_init, activity_regularizer=v_intra_act_fn)(v_spat_3)
    bn_6 = tf.keras.layers.BatchNormalization()(v_conv_6)
    v_add_1 = tf.keras.layers.Add()([bn_6, bn_2])
    v_act_1 = tf.keras.layers.Activation(activation=v_res_act_fn)(v_add_1)

    v_conv_5 = tf.keras.layers.Conv3D(filters=v_conv_5_filters, kernel_size=v_conv_5_kernel, strides=(v_conv_5_strides_0, v_conv_5_strides_1, v_conv_5_strides_2), padding=v_conv_5_pad, data_format=channel_order, activation=activation_fn, kernel_initializer=kern_init, activity_regularizer=v_conv_regularizer)(v_act_1)
    bn_5 = tf.keras.layers.BatchNormalization()(v_conv_5)
    v_spat_3 = tf.keras.layers.SpatialDropout3D(rate=v_spatial_drop_rate_3)(bn_5)
    v_conv_6 = tf.keras.layers.Conv3D(filters=v_conv_6_filters, kernel_size=v_conv_6_kernel, strides=(v_conv_6_strides_0, v_conv_6_strides_1, v_conv_6_strides_2), padding=v_conv_6_pad, data_format=channel_order, activation=activation_fn, kernel_initializer=kern_init, activity_regularizer=v_intra_act_fn)(v_spat_3)
    bn_6 = tf.keras.layers.BatchNormalization()(v_conv_6)
    v_add_1 = tf.keras.layers.Add()([bn_6, v_act_1])
    v_act_1 = tf.keras.layers.Activation(activation=v_res_act_fn)(v_add_1)

    v_conv_5 = tf.keras.layers.Conv3D(filters=v_conv_5_filters, kernel_size=v_conv_5_kernel, strides=(v_conv_5_strides_0, v_conv_5_strides_1, v_conv_5_strides_2), padding=v_conv_5_pad, data_format=channel_order, activation=activation_fn, kernel_initializer=kern_init, activity_regularizer=v_conv_regularizer)(v_act_1)
    bn_5 = tf.keras.layers.BatchNormalization()(v_conv_5)
    v_spat_3 = tf.keras.layers.SpatialDropout3D(rate=v_spatial_drop_rate_3)(bn_5)
    v_conv_6 = tf.keras.layers.Conv3D(filters=v_conv_6_filters, kernel_size=v_conv_6_kernel, strides=(v_conv_6_strides_0, v_conv_6_strides_1, v_conv_6_strides_2), padding=v_conv_6_pad, data_format=channel_order, activation=activation_fn, kernel_initializer=kern_init, activity_regularizer=v_intra_act_fn)(v_spat_3)
    bn_6 = tf.keras.layers.BatchNormalization()(v_conv_6)
    v_add_1 = tf.keras.layers.Add()([bn_6, v_act_1])
    v_act_1 = tf.keras.layers.Activation(activation=v_res_act_fn)(v_add_1)

    v_conv_9 = tf.keras.layers.Conv3D(filters=v_conv_9_filters, kernel_size=v_conv_9_kernel, strides=(v_conv_9_strides_0, v_conv_9_strides_1, v_conv_9_strides_2), padding=v_conv_9_pad, data_format=channel_order, activation=activation_fn, kernel_initializer=kern_init, activity_regularizer=v_conv_regularizer)(v_act_1)
    bn_9 = tf.keras.layers.BatchNormalization()(v_conv_9)
    v_spat_5 = tf.keras.layers.SpatialDropout3D(rate=v_spatial_drop_rate_5)(bn_9)
    v_conv_10 = tf.keras.layers.Conv3D(filters=v_conv_10_filters, kernel_size=v_conv_10_kernel, strides=(v_conv_10_strides_0, v_conv_10_strides_1, v_conv_10_strides_2), padding=v_conv_10_pad, data_format=channel_order, activation=activation_fn, kernel_initializer=kern_init, activity_regularizer=v_intra_act_fn)(v_spat_5)
    bn_10 = tf.keras.layers.BatchNormalization()(v_conv_10)
    v_conv_11 = tf.keras.layers.Conv3D(filters=v_conv_11_filters, kernel_size=v_conv_11_kernel, strides=(v_conv_11_strides_0, v_conv_11_strides_1, v_conv_11_strides_2), padding=v_conv_11_pad, data_format=channel_order, activation=activation_fn, kernel_initializer=kern_init, activity_regularizer=v_intra_act_fn)(v_act_1)
    bn_11 = tf.keras.layers.BatchNormalization()(v_conv_11)
    v_add_3 = tf.keras.layers.Add()([bn_10, bn_11])
    v_act_3 = tf.keras.layers.Activation(activation=v_res_act_fn)(v_add_3)

    v_conv_9 = tf.keras.layers.Conv3D(filters=v_conv_9_filters, kernel_size=v_conv_9_kernel, strides=(v_conv_9_strides_0, v_conv_9_strides_1, v_conv_9_strides_2), padding=v_conv_9_pad, data_format=channel_order, activation=activation_fn, kernel_initializer=kern_init, activity_regularizer=v_conv_regularizer)(v_act_3)
    bn_9 = tf.keras.layers.BatchNormalization()(v_conv_9)
    v_spat_5 = tf.keras.layers.SpatialDropout3D(rate=v_spatial_drop_rate_5)(bn_9)
    v_conv_10 = tf.keras.layers.Conv3D(filters=v_conv_10_filters, kernel_size=v_conv_10_kernel, strides=(v_conv_10_strides_0, v_conv_10_strides_1, v_conv_10_strides_2), padding=v_conv_10_pad, data_format=channel_order, activation=activation_fn, kernel_initializer=kern_init, activity_regularizer=v_intra_act_fn)(v_spat_5)
    bn_10 = tf.keras.layers.BatchNormalization()(v_conv_10)
    v_conv_11 = tf.keras.layers.Conv3D(filters=v_conv_11_filters, kernel_size=v_conv_11_kernel, strides=(v_conv_11_strides_0, v_conv_11_strides_1, v_conv_11_strides_2), padding=v_conv_11_pad, data_format=channel_order, activation=activation_fn, kernel_initializer=kern_init, activity_regularizer=v_intra_act_fn)(v_act_3)
    bn_11 = tf.keras.layers.BatchNormalization()(v_conv_11)
    v_add_3 = tf.keras.layers.Add()([bn_10, bn_11])
    v_act_3 = tf.keras.layers.Activation(activation=v_res_act_fn)(v_add_3)

    v_conv_9 = tf.keras.layers.Conv3D(filters=v_conv_9_filters, kernel_size=v_conv_9_kernel, strides=(v_conv_9_strides_0, v_conv_9_strides_1, v_conv_9_strides_2), padding=v_conv_9_pad, data_format=channel_order, activation=activation_fn, kernel_initializer=kern_init, activity_regularizer=v_conv_regularizer)(v_act_3)
    bn_9 = tf.keras.layers.BatchNormalization()(v_conv_9)
    v_spat_5 = tf.keras.layers.SpatialDropout3D(rate=v_spatial_drop_rate_5)(bn_9)
    v_conv_10 = tf.keras.layers.Conv3D(filters=v_conv_10_filters, kernel_size=v_conv_10_kernel, strides=(v_conv_10_strides_0, v_conv_10_strides_1, v_conv_10_strides_2), padding=v_conv_10_pad, data_format=channel_order, activation=activation_fn, kernel_initializer=kern_init, activity_regularizer=v_intra_act_fn)(v_spat_5)
    bn_10 = tf.keras.layers.BatchNormalization()(v_conv_10)
    v_conv_11 = tf.keras.layers.Conv3D(filters=v_conv_11_filters, kernel_size=v_conv_11_kernel, strides=(v_conv_11_strides_0, v_conv_11_strides_1, v_conv_11_strides_2), padding=v_conv_11_pad, data_format=channel_order, activation=activation_fn, kernel_initializer=kern_init, activity_regularizer=v_intra_act_fn)(v_act_3)
    bn_11 = tf.keras.layers.BatchNormalization()(v_conv_11)
    v_add_3 = tf.keras.layers.Add()([bn_10, bn_11])
    v_act_3 = tf.keras.layers.Activation(activation=v_res_act_fn)(v_add_3)


    bn_18 = tf.keras.layers.BatchNormalization()(v_act_3)
    v_spat_8 = tf.keras.layers.SpatialDropout3D(rate=v_spatial_drop_rate_8)(bn_18)
    v_conv_18 = tf.keras.layers.Conv3D(filters=v_conv_18_filters, kernel_size=v_conv_18_kernel, strides=(v_conv_18_strides_0, v_conv_18_strides_1, v_conv_18_strides_2), padding=v_conv_18_pad, data_format=channel_order, activation=activation_fn, kernel_initializer=kern_init, activity_regularizer=v_conv_regularizer)(v_spat_8)

    v_flatten_0 = tf.keras.layers.Flatten()(v_conv_18)

    bn_19 = tf.keras.layers.BatchNormalization()(v_flatten_0)
    dense_1_v = tf.keras.layers.Dense(units=dense_1_v_units, activation=activation_fn, kernel_initializer=kern_init, activity_regularizer=dense_regularizer_1)(bn_19)

    bn_20 = tf.keras.layers.BatchNormalization()(dense_1_v)
    dense_2_v = tf.keras.layers.Dense(units=dense_2_v_units, activation=activation_fn, kernel_initializer=kern_init, activity_regularizer=dense_regularizer_1)(bn_20)

    bn_21_v = tf.keras.layers.BatchNormalization()(dense_2_v)


    per_image_stand_1 = tf.keras.layers.Lambda(lambda frame: tf.image.per_image_standardization(frame))(input_fluoro_1)

    bn_0 = tf.keras.layers.BatchNormalization()(per_image_stand_1)
    conv_0_1 = tf.keras.layers.Conv2D(filters=conv_0_filters, kernel_size=conv_0_kernel, strides=conv_0_strides, padding=conv_0_pad, data_format=channel_order, activation=activation_fn, kernel_initializer=kern_init, activity_regularizer=conv_regularizer)(bn_0)

    bn_1 = tf.keras.layers.BatchNormalization()(conv_0_1)
    spat_0_1 = tf.keras.layers.SpatialDropout2D(rate=spatial_drop_rate_0)(bn_1)
    conv_1_1 = tf.keras.layers.Conv2D(filters=conv_1_filters, kernel_size=conv_1_kernel, strides=conv_1_strides, padding=conv_1_pad, data_format=channel_order, activation=activation_fn, kernel_initializer=kern_init, activity_regularizer=conv_regularizer)(spat_0_1)


    pool_0_1 = tf.keras.layers.AveragePooling2D(pool_size=pool_0_size, padding=pool_0_pad)(conv_1_1)

    bn_2 = tf.keras.layers.BatchNormalization()(pool_0_1)


    conv_2_1 = tf.keras.layers.Conv2D(filters=conv_2_filters, kernel_size=conv_2_kernel, strides=conv_2_strides, padding=conv_2_pad, data_format=channel_order, activation=activation_fn, kernel_initializer=kern_init, activity_regularizer=conv_regularizer)(bn_2)
    bn_3 = tf.keras.layers.BatchNormalization()(conv_2_1)
    spat_1_1 = tf.keras.layers.SpatialDropout2D(rate=spatial_drop_rate_1)(bn_3)
    conv_3_1 = tf.keras.layers.Conv2D(filters=conv_3_filters, kernel_size=conv_3_kernel, strides=conv_3_strides, padding=conv_3_pad, data_format=channel_order, activation=intra_act_fn, kernel_initializer=kern_init, activity_regularizer=conv_regularizer)(spat_1_1)
    bn_4 = tf.keras.layers.BatchNormalization()(conv_3_1)
    add_0 = tf.keras.layers.Add()([bn_4, bn_2])
    act_0 = tf.keras.layers.Activation(activation=res_act_fn)(add_0)


    conv_2_1 = tf.keras.layers.Conv2D(filters=conv_2_filters, kernel_size=conv_2_kernel, strides=conv_2_strides, padding=conv_2_pad, data_format=channel_order, activation=activation_fn, kernel_initializer=kern_init, activity_regularizer=conv_regularizer)(act_0)
    bn_3 = tf.keras.layers.BatchNormalization()(conv_2_1)
    spat_1_1 = tf.keras.layers.SpatialDropout2D(rate=spatial_drop_rate_1)(bn_3)
    conv_3_1 = tf.keras.layers.Conv2D(filters=conv_3_filters, kernel_size=conv_3_kernel, strides=conv_3_strides, padding=conv_3_pad, data_format=channel_order, activation=intra_act_fn, kernel_initializer=kern_init, activity_regularizer=conv_regularizer)(spat_1_1)
    bn_4 = tf.keras.layers.BatchNormalization()(conv_3_1)
    add_0 = tf.keras.layers.Add()([bn_4, act_0])
    act_0 = tf.keras.layers.Activation(activation=res_act_fn)(add_0)

    conv_2_1 = tf.keras.layers.Conv2D(filters=conv_2_filters, kernel_size=conv_2_kernel, strides=conv_2_strides, padding=conv_2_pad, data_format=channel_order, activation=activation_fn, kernel_initializer=kern_init, activity_regularizer=conv_regularizer)(act_0)
    bn_3 = tf.keras.layers.BatchNormalization()(conv_2_1)
    spat_1_1 = tf.keras.layers.SpatialDropout2D(rate=spatial_drop_rate_1)(bn_3)
    conv_3_1 = tf.keras.layers.Conv2D(filters=conv_3_filters, kernel_size=conv_3_kernel, strides=conv_3_strides, padding=conv_3_pad, data_format=channel_order, activation=intra_act_fn, kernel_initializer=kern_init, activity_regularizer=conv_regularizer)(spat_1_1)
    bn_4 = tf.keras.layers.BatchNormalization()(conv_3_1)
    add_0 = tf.keras.layers.Add()([bn_4, act_0])
    act_0_1 = tf.keras.layers.Activation(activation=res_act_fn)(add_0)

    conv_2_1 = tf.keras.layers.Conv2D(filters=conv_2_filters, kernel_size=conv_2_kernel, strides=conv_2_strides, padding=conv_2_pad, data_format=channel_order, activation=activation_fn, kernel_initializer=kern_init, activity_regularizer=conv_regularizer)(act_0)
    bn_3 = tf.keras.layers.BatchNormalization()(conv_2_1)
    spat_1_1 = tf.keras.layers.SpatialDropout2D(rate=spatial_drop_rate_1)(bn_3)
    conv_3_1 = tf.keras.layers.Conv2D(filters=conv_3_filters, kernel_size=conv_3_kernel, strides=conv_3_strides, padding=conv_3_pad, data_format=channel_order, activation=intra_act_fn, kernel_initializer=kern_init, activity_regularizer=conv_regularizer)(spat_1_1)
    bn_4 = tf.keras.layers.BatchNormalization()(conv_3_1)
    add_0 = tf.keras.layers.Add()([bn_4, act_0])
    act_0 = tf.keras.layers.Activation(activation=res_act_fn)(add_0)


    conv_2_1 = tf.keras.layers.Conv2D(filters=conv_2_filters, kernel_size=conv_2_kernel, strides=conv_2_strides, padding=conv_2_pad, data_format=channel_order, activation=activation_fn, kernel_initializer=kern_init, activity_regularizer=conv_regularizer)(act_0)
    bn_3 = tf.keras.layers.BatchNormalization()(conv_2_1)
    spat_1_1 = tf.keras.layers.SpatialDropout2D(rate=spatial_drop_rate_1)(bn_3)
    conv_3_1 = tf.keras.layers.Conv2D(filters=conv_3_filters, kernel_size=conv_3_kernel, strides=conv_3_strides, padding=conv_3_pad, data_format=channel_order, activation=intra_act_fn, kernel_initializer=kern_init, activity_regularizer=conv_regularizer)(spat_1_1)
    bn_4 = tf.keras.layers.BatchNormalization()(conv_3_1)
    add_0 = tf.keras.layers.Add()([bn_4, act_0])
    act_0 = tf.keras.layers.Activation(activation=res_act_fn)(add_0)

    conv_2_1 = tf.keras.layers.Conv2D(filters=conv_2_filters, kernel_size=conv_2_kernel, strides=conv_2_strides, padding=conv_2_pad, data_format=channel_order, activation=activation_fn, kernel_initializer=kern_init, activity_regularizer=conv_regularizer)(act_0)
    bn_3 = tf.keras.layers.BatchNormalization()(conv_2_1)
    spat_1_1 = tf.keras.layers.SpatialDropout2D(rate=spatial_drop_rate_1)(bn_3)
    conv_3_1 = tf.keras.layers.Conv2D(filters=conv_3_filters, kernel_size=conv_3_kernel, strides=conv_3_strides, padding=conv_3_pad, data_format=channel_order, activation=intra_act_fn, kernel_initializer=kern_init, activity_regularizer=conv_regularizer)(spat_1_1)
    bn_4 = tf.keras.layers.BatchNormalization()(conv_3_1)
    add_0 = tf.keras.layers.Add()([bn_4, act_0])
    act_0 = tf.keras.layers.Activation(activation=res_act_fn)(add_0)


    per_image_stand_2 = tf.keras.layers.Lambda(lambda frame: tf.image.per_image_standardization(frame))(input_fluoro_2)

    bn_0 = tf.keras.layers.BatchNormalization()(per_image_stand_2)
    conv_0_1 = tf.keras.layers.Conv2D(filters=conv_0_filters, kernel_size=conv_0_kernel, strides=conv_0_strides, padding=conv_0_pad, data_format=channel_order, activation=activation_fn, kernel_initializer=kern_init, activity_regularizer=conv_regularizer)(bn_0)

    bn_1 = tf.keras.layers.BatchNormalization()(conv_0_1)
    spat_0_1 = tf.keras.layers.SpatialDropout2D(rate=spatial_drop_rate_0)(bn_1)
    conv_1_1 = tf.keras.layers.Conv2D(filters=conv_1_filters, kernel_size=conv_1_kernel, strides=conv_1_strides, padding=conv_1_pad, data_format=channel_order, activation=activation_fn, kernel_initializer=kern_init, activity_regularizer=conv_regularizer)(spat_0_1)


    pool_0_1 = tf.keras.layers.AveragePooling2D(pool_size=pool_0_size, padding=pool_0_pad)(conv_1_1)


    bn_2 = tf.keras.layers.BatchNormalization()(pool_0_1)

    conv_2_1 = tf.keras.layers.Conv2D(filters=conv_2_filters, kernel_size=conv_2_kernel, strides=conv_2_strides, padding=conv_2_pad, data_format=channel_order, activation=activation_fn, kernel_initializer=kern_init, activity_regularizer=conv_regularizer)(bn_2)
    bn_3 = tf.keras.layers.BatchNormalization()(conv_2_1)
    spat_1_1 = tf.keras.layers.SpatialDropout2D(rate=spatial_drop_rate_1)(bn_3)
    conv_3_1 = tf.keras.layers.Conv2D(filters=conv_3_filters, kernel_size=conv_3_kernel, strides=conv_3_strides, padding=conv_3_pad, data_format=channel_order, activation=intra_act_fn, kernel_initializer=kern_init, activity_regularizer=conv_regularizer)(spat_1_1)
    bn_4 = tf.keras.layers.BatchNormalization()(conv_3_1)
    add_0 = tf.keras.layers.Add()([bn_4, bn_2])
    act_0 = tf.keras.layers.Activation(activation=res_act_fn)(add_0)

    conv_2_1 = tf.keras.layers.Conv2D(filters=conv_2_filters, kernel_size=conv_2_kernel, strides=conv_2_strides, padding=conv_2_pad, data_format=channel_order, activation=activation_fn, kernel_initializer=kern_init, activity_regularizer=conv_regularizer)(act_0)
    bn_3 = tf.keras.layers.BatchNormalization()(conv_2_1)
    spat_1_1 = tf.keras.layers.SpatialDropout2D(rate=spatial_drop_rate_1)(bn_3)
    conv_3_1 = tf.keras.layers.Conv2D(filters=conv_3_filters, kernel_size=conv_3_kernel, strides=conv_3_strides, padding=conv_3_pad, data_format=channel_order, activation=intra_act_fn, kernel_initializer=kern_init, activity_regularizer=conv_regularizer)(spat_1_1)
    bn_4 = tf.keras.layers.BatchNormalization()(conv_3_1)
    add_0 = tf.keras.layers.Add()([bn_4, act_0])
    act_0 = tf.keras.layers.Activation(activation=res_act_fn)(add_0)

    conv_2_1 = tf.keras.layers.Conv2D(filters=conv_2_filters, kernel_size=conv_2_kernel, strides=conv_2_strides, padding=conv_2_pad, data_format=channel_order, activation=activation_fn, kernel_initializer=kern_init, activity_regularizer=conv_regularizer)(act_0)
    bn_3 = tf.keras.layers.BatchNormalization()(conv_2_1)
    spat_1_1 = tf.keras.layers.SpatialDropout2D(rate=spatial_drop_rate_1)(bn_3)
    conv_3_1 = tf.keras.layers.Conv2D(filters=conv_3_filters, kernel_size=conv_3_kernel, strides=conv_3_strides, padding=conv_3_pad, data_format=channel_order, activation=intra_act_fn, kernel_initializer=kern_init, activity_regularizer=conv_regularizer)(spat_1_1)
    bn_4 = tf.keras.layers.BatchNormalization()(conv_3_1)
    add_0 = tf.keras.layers.Add()([bn_4, act_0])
    act_0_1 = tf.keras.layers.Activation(activation=res_act_fn)(add_0)

    conv_2_1 = tf.keras.layers.Conv2D(filters=conv_2_filters, kernel_size=conv_2_kernel, strides=conv_2_strides, padding=conv_2_pad, data_format=channel_order, activation=activation_fn, kernel_initializer=kern_init, activity_regularizer=conv_regularizer)(act_0)
    bn_3 = tf.keras.layers.BatchNormalization()(conv_2_1)
    spat_1_1 = tf.keras.layers.SpatialDropout2D(rate=spatial_drop_rate_1)(bn_3)
    conv_3_1 = tf.keras.layers.Conv2D(filters=conv_3_filters, kernel_size=conv_3_kernel, strides=conv_3_strides, padding=conv_3_pad, data_format=channel_order, activation=intra_act_fn, kernel_initializer=kern_init, activity_regularizer=conv_regularizer)(spat_1_1)
    bn_4 = tf.keras.layers.BatchNormalization()(conv_3_1)
    add_0 = tf.keras.layers.Add()([bn_4, act_0])
    act_0 = tf.keras.layers.Activation(activation=res_act_fn)(add_0)

    conv_2_1 = tf.keras.layers.Conv2D(filters=conv_2_filters, kernel_size=conv_2_kernel, strides=conv_2_strides, padding=conv_2_pad, data_format=channel_order, activation=activation_fn, kernel_initializer=kern_init, activity_regularizer=conv_regularizer)(act_0)
    bn_3 = tf.keras.layers.BatchNormalization()(conv_2_1)
    spat_1_1 = tf.keras.layers.SpatialDropout2D(rate=spatial_drop_rate_1)(bn_3)
    conv_3_1 = tf.keras.layers.Conv2D(filters=conv_3_filters, kernel_size=conv_3_kernel, strides=conv_3_strides, padding=conv_3_pad, data_format=channel_order, activation=intra_act_fn, kernel_initializer=kern_init, activity_regularizer=conv_regularizer)(spat_1_1)
    bn_4 = tf.keras.layers.BatchNormalization()(conv_3_1)
    add_0 = tf.keras.layers.Add()([bn_4, act_0])
    act_0 = tf.keras.layers.Activation(activation=res_act_fn)(add_0)

    conv_2_1 = tf.keras.layers.Conv2D(filters=conv_2_filters, kernel_size=conv_2_kernel, strides=conv_2_strides, padding=conv_2_pad, data_format=channel_order, activation=activation_fn, kernel_initializer=kern_init, activity_regularizer=conv_regularizer)(act_0)
    bn_3 = tf.keras.layers.BatchNormalization()(conv_2_1)
    spat_1_1 = tf.keras.layers.SpatialDropout2D(rate=spatial_drop_rate_1)(bn_3)
    conv_3_1 = tf.keras.layers.Conv2D(filters=conv_3_filters, kernel_size=conv_3_kernel, strides=conv_3_strides, padding=conv_3_pad, data_format=channel_order, activation=intra_act_fn, kernel_initializer=kern_init, activity_regularizer=conv_regularizer)(spat_1_1)
    bn_4 = tf.keras.layers.BatchNormalization()(conv_3_1)
    add_0 = tf.keras.layers.Add()([bn_4, act_0])
    act_0_2 = tf.keras.layers.Activation(activation=res_act_fn)(add_0)


    comb_fluoro_0 = tf.keras.layers.concatenate([act_0_1, act_0_2])

    comb_0 = tf.keras.layers.Conv2D(filters=comb_0_filters, kernel_size=comb_0_kernel, strides=comb_0_strides, padding=comb_0_pad, data_format=channel_order, activation=activation_fn, kernel_initializer=kern_init, activity_regularizer=conv_regularizer)(comb_fluoro_0)
    bn_0 = tf.keras.layers.BatchNormalization()(comb_0)
    spat_0 = tf.keras.layers.SpatialDropout2D(rate=comb_spatial_0)(bn_0)
    comb_1 = tf.keras.layers.Conv2D(filters=comb_1_filters, kernel_size=comb_1_kernel, strides=comb_1_strides, padding=comb_1_pad, data_format=channel_order, activation=c_intra_act_fn, kernel_initializer=kern_init, activity_regularizer=conv_regularizer)(spat_0)
    bn_1 = tf.keras.layers.BatchNormalization()(comb_1)
    add_0 = tf.keras.layers.Add()([comb_fluoro_0, bn_1])
    act_0 = tf.keras.layers.Activation(activation=c_res_act_fn)(add_0)


    comb_0 = tf.keras.layers.Conv2D(filters=comb_0_filters, kernel_size=comb_0_kernel, strides=comb_0_strides, padding=comb_0_pad, data_format=channel_order, activation=activation_fn, kernel_initializer=kern_init, activity_regularizer=conv_regularizer)(act_0)
    bn_0 = tf.keras.layers.BatchNormalization()(comb_0)
    spat_0 = tf.keras.layers.SpatialDropout2D(rate=comb_spatial_0)(bn_0)
    comb_1 = tf.keras.layers.Conv2D(filters=comb_1_filters, kernel_size=comb_1_kernel, strides=comb_1_strides, padding=comb_1_pad, data_format=channel_order, activation=c_intra_act_fn, kernel_initializer=kern_init, activity_regularizer=conv_regularizer)(spat_0)
    bn_1 = tf.keras.layers.BatchNormalization()(comb_1)
    add_0 = tf.keras.layers.Add()([act_0, bn_1])
    act_0 = tf.keras.layers.Activation(activation=c_res_act_fn)(add_0)

    comb_0 = tf.keras.layers.Conv2D(filters=comb_0_filters, kernel_size=comb_0_kernel, strides=comb_0_strides, padding=comb_0_pad, data_format=channel_order, activation=activation_fn, kernel_initializer=kern_init, activity_regularizer=conv_regularizer)(act_0)
    bn_0 = tf.keras.layers.BatchNormalization()(comb_0)
    spat_0 = tf.keras.layers.SpatialDropout2D(rate=comb_spatial_0)(bn_0)
    comb_1 = tf.keras.layers.Conv2D(filters=comb_1_filters, kernel_size=comb_1_kernel, strides=comb_1_strides, padding=comb_1_pad, data_format=channel_order, activation=c_intra_act_fn, kernel_initializer=kern_init, activity_regularizer=conv_regularizer)(spat_0)
    bn_1 = tf.keras.layers.BatchNormalization()(comb_1)
    add_0 = tf.keras.layers.Add()([act_0, bn_1])
    act_0 = tf.keras.layers.Activation(activation=c_res_act_fn)(add_0)

    comb_0 = tf.keras.layers.Conv2D(filters=comb_0_filters, kernel_size=comb_0_kernel, strides=comb_0_strides, padding=comb_0_pad, data_format=channel_order, activation=activation_fn, kernel_initializer=kern_init, activity_regularizer=conv_regularizer)(act_0)
    bn_0 = tf.keras.layers.BatchNormalization()(comb_0)
    spat_0 = tf.keras.layers.SpatialDropout2D(rate=comb_spatial_0)(bn_0)
    comb_1 = tf.keras.layers.Conv2D(filters=comb_1_filters, kernel_size=comb_1_kernel, strides=comb_1_strides, padding=comb_1_pad, data_format=channel_order, activation=c_intra_act_fn, kernel_initializer=kern_init, activity_regularizer=conv_regularizer)(spat_0)
    bn_1 = tf.keras.layers.BatchNormalization()(comb_1)
    add_0 = tf.keras.layers.Add()([act_0, bn_1])
    act_0 = tf.keras.layers.Activation(activation=c_res_act_fn)(add_0)

    comb_0 = tf.keras.layers.Conv2D(filters=comb_0_filters, kernel_size=comb_0_kernel, strides=comb_0_strides, padding=comb_0_pad, data_format=channel_order, activation=activation_fn, kernel_initializer=kern_init, activity_regularizer=conv_regularizer)(act_0)
    bn_0 = tf.keras.layers.BatchNormalization()(comb_0)
    spat_0 = tf.keras.layers.SpatialDropout2D(rate=comb_spatial_0)(bn_0)
    comb_1 = tf.keras.layers.Conv2D(filters=comb_1_filters, kernel_size=comb_1_kernel, strides=comb_1_strides, padding=comb_1_pad, data_format=channel_order, activation=c_intra_act_fn, kernel_initializer=kern_init, activity_regularizer=conv_regularizer)(spat_0)
    bn_1 = tf.keras.layers.BatchNormalization()(comb_1)
    add_0 = tf.keras.layers.Add()([act_0, bn_1])
    act_0 = tf.keras.layers.Activation(activation=c_res_act_fn)(add_0)

    comb_10 = tf.keras.layers.Conv2D(filters=comb_10_filters, kernel_size=comb_10_kernel, strides=comb_10_strides, padding=comb_10_pad, data_format=channel_order, activation=activation_fn, kernel_initializer=kern_init, activity_regularizer=conv_regularizer)(act_0)
    bn_10 = tf.keras.layers.BatchNormalization()(comb_10)
    spat_5 = tf.keras.layers.SpatialDropout2D(rate=comb_spatial_5)(bn_10)
    comb_11 = tf.keras.layers.Conv2D(filters=comb_11_filters, kernel_size=comb_11_kernel, strides=comb_11_strides, padding=comb_11_pad, data_format=channel_order, activation=c_intra_act_fn, kernel_initializer=kern_init, activity_regularizer=conv_regularizer)(spat_5)
    bn_11 = tf.keras.layers.BatchNormalization()(comb_11)
    comb_12 = tf.keras.layers.Conv2D(filters=comb_12_filters, kernel_size=comb_12_kernel, strides=comb_12_strides, padding=comb_12_pad, data_format=channel_order, activation=activation_fn, kernel_initializer=kern_init, activity_regularizer=conv_regularizer)(act_0)
    bn_12 = tf.keras.layers.BatchNormalization()(comb_12)
    add_5 = tf.keras.layers.Add()([bn_11, bn_12])
    act_5 = tf.keras.layers.Activation(activation=c_res_act_fn)(add_5)

    comb_10 = tf.keras.layers.Conv2D(filters=comb_10_filters, kernel_size=comb_10_kernel, strides=comb_10_strides, padding=comb_10_pad, data_format=channel_order, activation=activation_fn, kernel_initializer=kern_init, activity_regularizer=conv_regularizer)(act_5)
    bn_10 = tf.keras.layers.BatchNormalization()(comb_10)
    spat_5 = tf.keras.layers.SpatialDropout2D(rate=comb_spatial_5)(bn_10)
    comb_11 = tf.keras.layers.Conv2D(filters=comb_11_filters, kernel_size=comb_11_kernel, strides=comb_11_strides, padding=comb_11_pad, data_format=channel_order, activation=c_intra_act_fn, kernel_initializer=kern_init, activity_regularizer=conv_regularizer)(spat_5)
    bn_11 = tf.keras.layers.BatchNormalization()(comb_11)
    comb_12 = tf.keras.layers.Conv2D(filters=comb_12_filters, kernel_size=comb_12_kernel, strides=comb_12_strides, padding=comb_12_pad, data_format=channel_order, activation=activation_fn, kernel_initializer=kern_init, activity_regularizer=conv_regularizer)(act_5)
    bn_12 = tf.keras.layers.BatchNormalization()(comb_12)
    add_5 = tf.keras.layers.Add()([bn_11, bn_12])
    act_5 = tf.keras.layers.Activation(activation=c_res_act_fn)(add_5)

    comb_10 = tf.keras.layers.Conv2D(filters=comb_10_filters, kernel_size=comb_10_kernel, strides=comb_10_strides, padding=comb_10_pad, data_format=channel_order, activation=activation_fn, kernel_initializer=kern_init, activity_regularizer=conv_regularizer)(act_5)
    bn_10 = tf.keras.layers.BatchNormalization()(comb_10)
    spat_5 = tf.keras.layers.SpatialDropout2D(rate=comb_spatial_5)(bn_10)
    comb_11 = tf.keras.layers.Conv2D(filters=comb_11_filters, kernel_size=comb_11_kernel, strides=comb_11_strides, padding=comb_11_pad, data_format=channel_order, activation=c_intra_act_fn, kernel_initializer=kern_init, activity_regularizer=conv_regularizer)(spat_5)
    bn_11 = tf.keras.layers.BatchNormalization()(comb_11)
    comb_12 = tf.keras.layers.Conv2D(filters=comb_12_filters, kernel_size=comb_12_kernel, strides=comb_12_strides, padding=comb_12_pad, data_format=channel_order, activation=activation_fn, kernel_initializer=kern_init, activity_regularizer=conv_regularizer)(act_5)
    bn_12 = tf.keras.layers.BatchNormalization()(comb_12)
    add_5 = tf.keras.layers.Add()([bn_11, bn_12])
    act_5 = tf.keras.layers.Activation(activation=c_res_act_fn)(add_5)

    comb_19 = tf.keras.layers.Conv2D(filters=comb_19_filters, kernel_size=comb_19_kernel, strides=comb_19_strides, padding=comb_19_pad, data_format=channel_order, activation=activation_fn, kernel_initializer=kern_init, activity_regularizer=conv_regularizer)(act_5)

    comb_flatten_1 = tf.keras.layers.Flatten()(comb_19)

    bn_19 = tf.keras.layers.BatchNormalization()(comb_flatten_1)
    dense_0_comb = tf.keras.layers.Dense(units=dense_comb_0_units, activation=activation_fn, kernel_initializer=kern_init, activity_regularizer=dense_regularizer_1)(bn_19)

    bn_20 = tf.keras.layers.BatchNormalization()(dense_0_comb)
    dense_1_comb = tf.keras.layers.Dense(units=dense_comb_1_units, activation=activation_fn, kernel_initializer=kern_init, activity_regularizer=dense_regularizer_1)(bn_20)

    bn_21_f = tf.keras.layers.BatchNormalization()(dense_1_comb)

    fluoro_vox_comb = tf.keras.layers.Add()([bn_21_f, bn_21_v])
    fluoro_vox_act = tf.keras.layers.Activation(activation=flu_vox_act_fn)(fluoro_vox_comb)

    bn_0 = tf.keras.layers.BatchNormalization()(fluoro_vox_act)
    vox_flu_0 = tf.keras.layers.Dense(units=vox_flu_units_0, activation=activation_fn, kernel_initializer=kern_init, activity_regularizer=dense_regularizer_1)(bn_0)

    bn_1 = tf.keras.layers.BatchNormalization()(vox_flu_0)
    vox_flu_1 = tf.keras.layers.Dense(units=vox_flu_units_1, activation=activation_fn, kernel_initializer=kern_init, activity_regularizer=dense_regularizer_1)(bn_1)

    bn_2 = tf.keras.layers.BatchNormalization()(vox_flu_1)
    vox_flu_2 = tf.keras.layers.Dense(units=vox_flu_units_2, activation=activation_fn, kernel_initializer=kern_init, activity_regularizer=dense_regularizer_1)(bn_2)

    bn_3 = tf.keras.layers.BatchNormalization()(vox_flu_2)
    vox_flu_3 = tf.keras.layers.Dense(units=vox_flu_units_3, activation=activation_fn, kernel_initializer=kern_init, activity_regularizer=dense_regularizer_1)(bn_3)

    bn_4 = tf.keras.layers.BatchNormalization()(vox_flu_3)
    vox_flu_4 = tf.keras.layers.Dense(units=vox_flu_units_4, activation=activation_fn, kernel_initializer=kern_init, activity_regularizer=dense_regularizer_1)(bn_4)

    bn_5_comb = tf.keras.layers.BatchNormalization()(vox_flu_4)

    bn_0 = tf.keras.layers.BatchNormalization()(input_cali)
    cali_0 = tf.keras.layers.Dense(units=cali_0_units, activation=activation_fn, kernel_initializer=kern_init, activity_regularizer=dense_regularizer_1)(bn_0)

    bn_1 = tf.keras.layers.BatchNormalization()(cali_0)
    cali_1 = tf.keras.layers.Dense(units=cali_1_units, activation=activation_fn, kernel_initializer=kern_init, activity_regularizer=dense_regularizer_1)(bn_1)

    bn_2 = tf.keras.layers.BatchNormalization()(cali_1)
    cali_2 = tf.keras.layers.Dense(units=cali_2_units, activation=activation_fn, kernel_initializer=kern_init, activity_regularizer=dense_regularizer_1)(bn_2)

    bn_3 = tf.keras.layers.BatchNormalization()(cali_2)
    cali_3 = tf.keras.layers.Dense(units=cali_3_units, activation=activation_fn, kernel_initializer=kern_init, activity_regularizer=dense_regularizer_1)(bn_3)

    bn_4_c = tf.keras.layers.BatchNormalization()(cali_3)

    top_level_comb = tf.keras.layers.Add()([bn_4_c, bn_5_comb])
    top_level_act = tf.keras.layers.Activation(activation=top_level_act_fn)(top_level_comb)

    top_dense_0 = tf.keras.layers.Dense(units=top_dense_0, activation=activation_fn, kernel_initializer=kern_init, activity_regularizer=dense_regularizer_2)(top_level_act)
    bn_0 = tf.keras.layers.BatchNormalization()(top_dense_0)
    top_dense_1 = tf.keras.layers.Dense(units=top_dense_1, activation=top_level_intra, kernel_initializer=kern_init, activity_regularizer=dense_regularizer_2)(bn_0)
    bn_1 = tf.keras.layers.BatchNormalization()(top_dense_1)
    add_0 = tf.keras.layers.Add()([bn_1, bn_4_c])
    act_0 = tf.keras.layers.Activation(activation=top_level_act_fn)(add_0)


    top_dense_2 = tf.keras.layers.Dense(units=top_dense_2, activation=activation_fn, kernel_initializer=kern_init, activity_regularizer=dense_regularizer_2)(act_0)
    bn_2 = tf.keras.layers.BatchNormalization()(top_dense_2)
    top_dense_3 = tf.keras.layers.Dense(units=top_dense_3, activation=top_level_intra, kernel_initializer=kern_init, activity_regularizer=dense_regularizer_2)(bn_2)
    bn_3 = tf.keras.layers.BatchNormalization()(top_dense_3)
    add_1 = tf.keras.layers.Add()([bn_3, act_0])
    act_1 = tf.keras.layers.Activation(activation=top_level_act_fn)(add_1)


    top_dense_4 = tf.keras.layers.Dense(units=top_dense_2, activation=activation_fn, kernel_initializer=kern_init, activity_regularizer=dense_regularizer_2)(act_1)
    bn_4 = tf.keras.layers.BatchNormalization()(top_dense_4)
    top_dense_5 = tf.keras.layers.Dense(units=top_dense_3, activation=top_level_intra, kernel_initializer=kern_init, activity_regularizer=dense_regularizer_2)(bn_4)
    bn_5 = tf.keras.layers.BatchNormalization()(top_dense_5)
    add_2 = tf.keras.layers.Add()([bn_5, act_1])
    act_2 = tf.keras.layers.Activation(activation=top_level_act_fn)(add_2)



    top_dense_6 = tf.keras.layers.Dense(units=top_dense_4, activation=activation_fn, kernel_initializer=kern_init, activity_regularizer=None)(act_2)

    main_output = tf.keras.layers.Dense(units=main_output_units, activation=main_output_act, kernel_initializer=kern_init, name='main_output')(top_dense_6)

    model = tf.keras.Model(inputs=[input_vox, input_fluoro_1, input_fluoro_2, input_cali], outputs=main_output)

    model.compile(optimizer=model_opt, loss=model_loss, metrics=[model_metric])

    result = model.fit(x={'input_vox': np.expand_dims(vox_mat_train, axis=-1), 'input_fluoro_1': np.expand_dims(image_mat_train[:, 0, :, :], axis=-1), 'input_fluoro_2': np.expand_dims(image_mat_train[:, 1, :, :], axis=-1), 'input_cali': cali_mat_train}, y=label_mat_train, validation_split=0.2, epochs=model_epochs, batch_size=model_batchsize, shuffle=True, verbose=False)

    return {'loss': np.amin(result.history['loss']), 'status': STATUS_OK, 'model': model}


best_run, best_model = optim.minimize(model=fluoro_model, data=data_comp, algo=tpe.suggest, max_evals=3, trials=Trials())

json1 = json.dumps(best_run)

f = open(os.path.abspath(os.path.join(save_dir, 'best_run_hyperas.json')), 'w')
f.write(json1)
f.close()


w = csv.writer(open(os.path.abspath(os.path.join(save_dir, 'best_run_hyperas.csv')), 'w'))
for key, val in best_run.items():
    w.writerow([key, val])





best_model.save(os.path.abspath(os.path.join(save_dir, 'vox_fluoro_img_stnd_hyperas' + '_' + 'best_model_hyperas.h5')))







