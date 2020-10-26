#coding=utf-8

try:
    import numpy as np
except:
    pass

try:
    import h5py
except:
    pass

try:
    import tensorflow as tf
except:
    pass

try:
    import keras
except:
    pass

try:
    import os
except:
    pass

try:
    import graphviz
except:
    pass

try:
    import sys
except:
    pass

try:
    import hyperas
except:
    pass

try:
    import hyperopt
except:
    pass

try:
    from sklearn.model_selection import train_test_split
except:
    pass

try:
    import json
except:
    pass

try:
    import csv
except:
    pass

try:
    from hyperopt import Trials, STATUS_OK, tpe
except:
    pass

try:
    from hyperas import optim
except:
    pass

try:
    from hyperas.distributions import choice, uniform
except:
    pass
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

image_file = h5py.File(os.path.expanduser('~/fluoro/data/compilation/images.h5py'), 'r')
image_init = image_file['image_dset']
image_mat = image_init[:]

label_file = h5py.File(os.path.expanduser('~/fluoro/data/compilation/labels.h5py'), 'r')
label_init = label_file['labels_dset']
label_mat = label_init[:]

cali_file = h5py.File(os.path.expanduser('~/fluoro/data/compilation/calibration.h5py'), 'r')
cali_init = cali_file['cali_len3_rot']
cali_mat = cali_init[:]

image_train_cum, image_test, cali_train_cum, cali_test, label_train_cum, label_test = train_test_split(image_mat, cali_mat, label_mat, shuffle=True, test_size=0.2)


def keras_fmin_fnct(space):

    def root_mean_squared_error(y_true, y_pred):
        return keras.backend.sqrt(keras.backend.mean(keras.backend.square(y_pred - y_true)))
    channel_order = 'channels_last'
    img_input_shape = (128, 128, 1)

    regularizer = keras.regularizers.l1_l2(l1=space['l1'], l2=space['l1_1'])
    activation_fn = space['activation_fn']

    kern_init = space['kern_init']

    conv_1_filters = space['conv_1_filters']
    conv_1_kernel = space['conv_1_kernel']
    conv_1_strides = space['conv_1_strides']
    conv_1_padding = 'valid'

    spatial_drop_rate_1 = space['l1_2']

    pool_1_size = space['pool_1_size']
    pool_1_padding = 'same'

    conv_2_filters = space['conv_2_filters']
    conv_2_kernel = space['conv_2_kernel']
    conv_2_strides = space['conv_1_strides_1']
    conv_2_padding = 'same'

    pool_2_size = space['pool_1_size_1']
    pool_2_padding = 'same'

    conv_3_filters = space['conv_3_filters']
    conv_3_kernel = space['pool_1_size_2']
    conv_3_strides = space['conv_1_strides_2']
    conv_3_padding = 'valid'

    pool_3_size = (2, 2)
    pool_3_padding = 'valid'

    dense_1_f_units = space['dense_1_f_units']
    dense_1_f_bias = True

    dense_2_f_units = space['dense_1_f_units_1']
    dense_2_f_bias = True

    dense_3_f_units = space['dense_1_f_units_2']
    dense_3_f_bias = True

    dense_1_ca_units = space['dense_1_ca_units']
    dense_1_ca_bias = True

    dense_2_co_units = space['conv_2_filters_1']
    dense_2_co_bias = True

    drop_1_comb_rate = space['l1_3']

    dense_3_co_units = space['conv_2_filters_2']
    dense_3_co_bias = True

    main_output_units = 6
    main_output_act = 'linear'
    main_output_bias = True

    model_opt = space['model_opt']
    model_loss = 'mse'
    model_metric = root_mean_squared_error

    model_epochs = space['model_epochs']
    model_batchsize = space['model_batchsize']

    input_fluoro_1 = keras.Input(shape=img_input_shape, dtype=np.dtype('float32'), name='fluoro1_inpt')
    input_fluoro_2 = keras.Input(shape=img_input_shape, dtype=np.dtype('float32'), name='fluoro2_inpt')
    input_cali = keras.Input(shape=(6,), dtype=np.dtype('float32'), name='cali_inpt')

    bn_1_1 = keras.layers.BatchNormalization()(input_fluoro_1)
    conv_1_1 = keras.layers.Conv2D(filters=conv_1_filters, kernel_size=conv_1_kernel, strides=conv_1_strides, padding=conv_1_padding, activation=activation_fn, input_shape=img_input_shape, data_format=channel_order, activity_regularizer=regularizer, kernel_initializer=kern_init)(bn_1_1)
    spat_1_1 = keras.layers.SpatialDropout2D(rate=spatial_drop_rate_1)(conv_1_1)
    pool_1_1 = keras.layers.MaxPooling2D(pool_size=pool_1_size, padding=pool_1_padding, data_format=channel_order)(spat_1_1)
    conv_2_1 = keras.layers.Conv2D(filters=conv_2_filters, kernel_size=conv_2_kernel, strides=conv_2_strides, padding=conv_2_padding, activation=activation_fn, data_format=channel_order, activity_regularizer=regularizer, kernel_initializer=kern_init)(pool_1_1)
    pool_2_1 = keras.layers.MaxPooling2D(pool_size=pool_2_size, padding=pool_2_padding, data_format=channel_order)(conv_2_1)
    conv_3_1 = keras.layers.Conv2D(filters=conv_3_filters, kernel_size=conv_3_kernel, strides=conv_3_strides, padding=conv_3_padding, data_format=channel_order, activity_regularizer=regularizer, kernel_initializer=kern_init)(pool_2_1)
    pool_3_1 = keras.layers.MaxPooling2D(pool_size=pool_3_size, padding=pool_3_padding, data_format=channel_order)(conv_3_1)
    flatten_1_1 = keras.layers.Flatten()(pool_3_1)
    dense_1_f_1 = keras.layers.Dense(units=dense_1_f_units, activation=activation_fn, use_bias=dense_1_f_bias, kernel_initializer=kern_init, activity_regularizer=regularizer, name='dense_1_f_1')(flatten_1_1)
    dense_2_f_1 = keras.layers.Dense(units=dense_2_f_units, activation=activation_fn, use_bias=dense_2_f_bias, kernel_initializer=kern_init, activity_regularizer=regularizer, name='dense_2_f_1')(dense_1_f_1)
    dense_3_f_1 = keras.layers.Dense(units=dense_3_f_units, activation=activation_fn, use_bias=dense_3_f_bias, kernel_initializer=kern_init, activity_regularizer=regularizer, name='dense_3_f_1')(dense_2_f_1)

    bn_1_2 = keras.layers.BatchNormalization()(input_fluoro_2)
    conv_1_2 = keras.layers.Conv2D(filters=conv_1_filters, kernel_size=conv_1_kernel, strides=conv_1_strides, padding=conv_1_padding, activation=activation_fn, input_shape=img_input_shape, data_format=channel_order, activity_regularizer=regularizer, kernel_initializer=kern_init)(bn_1_2)
    spat_1_2 = keras.layers.SpatialDropout2D(rate=spatial_drop_rate_1)(conv_1_2)
    pool_1_2 = keras.layers.MaxPooling2D(pool_size=pool_1_size, padding=pool_1_padding, data_format=channel_order)(spat_1_2)
    conv_2_2 = keras.layers.Conv2D(filters=conv_2_filters, kernel_size=conv_2_kernel, strides=conv_2_strides, padding=conv_2_padding, activation=activation_fn, data_format=channel_order, activity_regularizer=regularizer, kernel_initializer=kern_init)(pool_1_2)
    pool_2_2 = keras.layers.MaxPooling2D(pool_size=pool_2_size, padding=pool_2_padding, data_format=channel_order)(conv_2_2)
    conv_3_2 = keras.layers.Conv2D(filters=conv_3_filters, kernel_size=conv_3_kernel, strides=conv_3_strides, padding=conv_3_padding, data_format=channel_order, activity_regularizer=regularizer, kernel_initializer=kern_init)(pool_2_2)
    pool_3_2 = keras.layers.MaxPooling2D(pool_size=pool_3_size, padding=pool_3_padding, data_format=channel_order)(conv_3_2)
    flatten_1_2 = keras.layers.Flatten()(pool_3_2)
    dense_1_f_2 = keras.layers.Dense(units=dense_1_f_units, activation=activation_fn, use_bias=dense_1_f_bias, kernel_initializer=kern_init, activity_regularizer=regularizer, name='dense_1_f_2')(flatten_1_2)
    dense_2_f_2 = keras.layers.Dense(units=dense_2_f_units, activation=activation_fn, use_bias=dense_2_f_bias, kernel_initializer=kern_init, activity_regularizer=regularizer, name='dense_2_f_2')(dense_1_f_2)
    dense_3_f_2 = keras.layers.Dense(units=dense_3_f_units, activation=activation_fn, use_bias=dense_3_f_bias, kernel_initializer=kern_init, activity_regularizer=regularizer, name='dense_3_f_2')(dense_2_f_2)

    dense_1_cali = keras.layers.Dense(units=dense_1_ca_units, activation=activation_fn, use_bias=dense_1_ca_bias, kernel_initializer=kern_init, name='dense_1_cali')(input_cali)

    dense_1_comb = keras.layers.concatenate([dense_3_f_1, dense_3_f_2, dense_1_cali], name='dense_1_comb')

    dense_2_comb = keras.layers.Dense(units=dense_2_co_units, activation=activation_fn, use_bias=dense_2_co_bias, kernel_initializer=kern_init, name='dense_2_comb')(dense_1_comb)
    drop_1_comb = keras.layers.Dropout(rate=drop_1_comb_rate)(dense_2_comb)
    dense_3_comb = keras.layers.Dense(units=dense_3_co_units, activation=activation_fn, use_bias=dense_3_co_bias, kernel_initializer=kern_init, name='dense_3_comb')(drop_1_comb)
    main_output = keras.layers.Dense(units=main_output_units, activation=main_output_act, name='main_output')(dense_3_comb)

    model = keras.Model(inputs=[input_fluoro_1, input_fluoro_2, input_cali], outputs=main_output)

    keras.utils.plot_model(model, 'show.png', show_shapes=True)

    model.compile(optimizer=model_opt, loss=model_loss, metrics=[model_metric])

    result = model.fit(x=[np.expand_dims(image_train_cum[:, 0, :, :], axis=3), np.expand_dims(image_train_cum[:, 1, :, :], axis=3), cali_train_cum], y=label_train_cum, epochs=model_epochs, batch_size=model_batchsize, validation_split=0.2, shuffle=True, verbose=True)
    return {'loss': np.amin(result.history['loss']), 'status': STATUS_OK, 'model': model}

def get_space():
    return {
        'l1': hp.uniform('l1', 0, 1),
        'l1_1': hp.uniform('l1_1', 0, 1),
        'activation_fn': hp.choice('activation_fn', ['elu', 'relu']),
        'kern_init': hp.choice('kern_init', ['glorot_uniform', 'glorot_normal']),
        'conv_1_filters': hp.choice('conv_1_filters', [10, 20, 40, 50]),
        'conv_1_kernel': hp.choice('conv_1_kernel', [(10, 10), (5, 5), (3, 3)]),
        'conv_1_strides': hp.choice('conv_1_strides', [(2, 2), (1, 1)]),
        'l1_2': hp.uniform('l1_2', 0, 1),
        'pool_1_size': hp.choice('pool_1_size', [(2, 2), (3, 3)]),
        'conv_2_filters': hp.choice('conv_2_filters', [20, 40, 80]),
        'conv_2_kernel': hp.choice('conv_2_kernel', [(3, 3), (5, 5)]),
        'conv_1_strides_1': hp.choice('conv_1_strides_1', [(2, 2), (1, 1)]),
        'pool_1_size_1': hp.choice('pool_1_size_1', [(2, 2), (3, 3)]),
        'conv_3_filters': hp.choice('conv_3_filters', [20, 80, 100]),
        'pool_1_size_2': hp.choice('pool_1_size_2', [(2, 2), (3, 3)]),
        'conv_1_strides_2': hp.choice('conv_1_strides_2', [(2, 2), (1, 1)]),
        'dense_1_f_units': hp.choice('dense_1_f_units', [40, 80, 120]),
        'dense_1_f_units_1': hp.choice('dense_1_f_units_1', [40, 80, 120]),
        'dense_1_f_units_2': hp.choice('dense_1_f_units_2', [40, 80, 120]),
        'dense_1_ca_units': hp.choice('dense_1_ca_units', [6, 20, 60]),
        'conv_2_filters_1': hp.choice('conv_2_filters_1', [20, 40, 80]),
        'l1_3': hp.uniform('l1_3', 0, 1),
        'conv_2_filters_2': hp.choice('conv_2_filters_2', [20, 40, 80]),
        'model_opt': hp.choice('model_opt', ['adam', 'nadam', 'adagrad', 'rmsprop']),
        'model_epochs': hp.choice('model_epochs', [30, 40, 50]),
        'model_batchsize': hp.choice('model_batchsize', [5, 10, 30]),
    }
