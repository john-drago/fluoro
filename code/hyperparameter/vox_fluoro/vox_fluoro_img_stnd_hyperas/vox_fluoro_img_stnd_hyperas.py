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
save_dir = os.path.abspath(os.path.join(os.path.expanduser('~/fluoro/code/hyperparameter/vox_fluoro'), 'vox_fluoro_img_stnd_hyperas'))
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

    channel_order = 'channels_last'
    img_input_shape = (128, 128, 1)
    vox_input_shape = (199, 164, 566, 1)
    cali_input_shape = (6,)


    def cust_mean_squared_error_var(y_true, y_pred):
        base_dir = os.path.expanduser('~/fluoro/data/compilation')
        stats_file = h5py.File(os.path.join(base_dir, 'labels_stats.h5py'), 'r')
        var_dset = stats_file['var']
        var_v = var_dset[:]

        stats_file.close()

        return tf.keras.backend.mean(tf.keras.backend.square(y_pred - y_true) / var_v)





    v_conv_1_filters = {{choice([20, 30, 40])}}
    v_conv_1_kernel = {{choice([5, 7, 11, 13, 21])}}
    v_conv_1_strides = {{choice([1, 2])}}
    v_conv_1_pad = 'same'
    v_spatial_drop_rate_1 = {{uniform(0, 1)}}
    v_pool_1_size = {{choice([2, 3])}}
    v_pool_1_pad = 'same'
    v_conv_2_filters = {{choice([40, 50, 60, 80])}}
    v_conv_2_kernel = {{choice([5, 7, 11])}}
    v_conv_2_strides = {{choice([1, 2])}}
    v_conv_2_pad = 'same'
    v_spatial_drop_rate_2 = {{uniform(0, 1)}}
    v_pool_2_size = {{choice([2, 3])}}
    v_pool_2_pad = 'same'
    v_conv_3_filters = {{choice([40, 50, 60, 80])}}
    v_conv_3_kernel = {{choice([3, 5, 7])}}
    v_conv_3_strides = {{choice([1, 2])}}
    v_conv_3_pad = 'same'
    v_spatial_drop_rate_3 = {{uniform(0, 1)}}
    v_pool_3_size = {{choice([2, 3])}}
    v_pool_3_pad = 'same'
    dense_1_v_units = {{choice([750, 1000, 1500])}}
    dense_2_v_units = {{choice([500, 750, 1000])}}
    dense_3_v_units = {{choice([250, 500, 750])}}


    conv_1_filters = {{choice([20, 30, 40, 50, 60])}}
    conv_1_kernel = {{choice([3, 5, 7])}}
    conv_1_strides = {{choice([1, 2])}}
    conv_1_pad = 'same'
    spatial_drop_rate_1 = {{uniform(0, 1)}}
    pool_1_size = {{choice([2, 3])}}
    pool_1_pad = 'same'
    conv_2_filters = {{choice([40, 50, 60, 80])}}
    conv_2_kernel = {{choice([3, 5, 7])}}
    conv_2_strides = {{choice([1, 2])}}
    conv_2_pad = 'same'
    spatial_drop_rate_2 = {{uniform(0, 1)}}
    pool_2_size = {{choice([2, 3])}}
    pool_2_pad = 'same'
    conv_3_filters = {{choice([40, 50, 60, 80])}}
    conv_3_kernel = {{choice([3, 5, 7])}}
    conv_3_strides = {{choice([1, 2])}}
    conv_3_pad = 'same'
    pool_3_size = {{choice([2, 3])}}
    pool_3_pad = 'same'
    dense_1_f_units = {{choice([40, 60, 80])}}
    dense_2_f_units = {{choice([40, 60, 80])}}
    dense_3_f_units = {{choice([40, 60, 80])}}


    dense_1_cali_units = {{choice([10, 20, 30])}}
    dense_2_cali_units = {{choice([10, 20, 30])}}


    dense_1_co_units = {{choice([60, 80, 100, 200])}}
    drop_1_comb_rate = {{uniform(0, 1)}}
    dense_2_co_units = {{choice([20, 40, 60])}}
    dense_3_co_units = {{choice([20, 40, 60])}}
    dense_4_co_units = {{choice([20, 40, 60])}}


    main_output_units = 6
    main_output_act = 'linear'


    conv_regularizer = keras.regularizers.l1_l2(l1={{uniform(0, 1)}}, l2={{uniform(0, 1)}})
    dense_regularizer_1 = keras.regularizers.l1_l2(l1={{uniform(0, 1)}}, l2={{uniform(0, 1)}})
    dense_regularizer_2 = keras.regularizers.l1_l2(l1={{uniform(0, 1)}}, l2={{uniform(0, 1)}})
    activation_fn = {{choice(['elu', 'relu'])}}
    kern_init = {{choice(['glorot_uniform', 'glorot_normal'])}}
    model_opt = {{choice(['adam', 'nadam', 'adagrad', 'rmsprop'])}}
    model_epochs = {{choice([30, 40, 50])}}
    model_batchsize = 3
    model_loss = cust_mean_squared_error_var
    model_metric = cust_mean_squared_error_var


    input_vox = tf.keras.Input(shape=vox_input_shape, name='input_vox', dtype='float32')
    input_fluoro_1 = tf.keras.Input(shape=img_input_shape, name='input_fluoro_1', dtype='float32')
    input_fluoro_2 = tf.keras.Input(shape=img_input_shape, name='input_fluoro_2', dtype='float32')
    input_cali = tf.keras.Input(shape=cali_input_shape, name='input_cali', dtype='float32')

    v_conv_1 = tf.keras.layers.Conv3D(filters=v_conv_1_filters, kernel_size=v_conv_1_kernel, strides=v_conv_1_strides, padding=v_conv_1_pad, data_format=channel_order, activation=activation_fn, kernel_initializer=kern_init, activity_regularizer=conv_regularizer)(input_vox)
    v_spat_1 = tf.keras.layers.SpatialDropout3D(rate=v_spatial_drop_rate_1)(v_conv_1)
    v_pool_1 = tf.keras.layers.MaxPooling3D(pool_size=v_pool_1_size, padding=v_pool_1_pad, data_format=channel_order)(v_spat_1)


    v_bn_2 = tf.keras.layers.BatchNormalization()(v_pool_1)
    v_conv_2 = tf.keras.layers.Conv3D(filters=v_conv_2_filters, kernel_size=v_conv_2_kernel, strides=v_conv_2_strides, padding=v_conv_2_pad, data_format=channel_order, activation=activation_fn, kernel_initializer=kern_init, activity_regularizer=conv_regularizer)(v_bn_2)
    v_spat_2 = tf.keras.layers.SpatialDropout3D(rate=v_spatial_drop_rate_2)(v_conv_2)
    v_pool_2 = tf.keras.layers.MaxPooling3D(pool_size=v_pool_2_size, padding=v_pool_2_pad, data_format=channel_order)(v_spat_2)


    v_conv_3 = tf.keras.layers.Conv3D(filters=v_conv_3_filters, kernel_size=v_conv_3_kernel, strides=v_conv_3_strides, padding=v_conv_3_pad, data_format=channel_order, activation=activation_fn, kernel_initializer=kern_init, activity_regularizer=conv_regularizer)(v_pool_2)
    v_spat_3 = tf.keras.layers.SpatialDropout3D(rate=v_spatial_drop_rate_3)(v_conv_3)
    v_pool_3 = tf.keras.layers.MaxPooling3D(pool_size=v_pool_3_size, padding=v_pool_3_pad, data_format=channel_order)(v_spat_3)

    v_flatten_1 = tf.keras.layers.Flatten()(v_pool_3)


    dense_1_v = tf.keras.layers.Dense(units=dense_1_v_units, activation=activation_fn, kernel_initializer=kern_init, activity_regularizer=dense_regularizer_1)(v_flatten_1)
    dense_2_v = tf.keras.layers.Dense(units=dense_2_v_units, activation=activation_fn, kernel_initializer=kern_init, activity_regularizer=dense_regularizer_1)(dense_1_v)
    dense_3_v = tf.keras.layers.Dense(units=dense_3_v_units, activation=activation_fn, kernel_initializer=kern_init, activity_regularizer=dense_regularizer_1)(dense_2_v)


    per_image_stand_1 = tf.keras.layers.Lambda(lambda frame: tf.image.per_image_standardization(frame))(input_fluoro_1)
    conv_1_1 = tf.keras.layers.Conv2D(filters=conv_1_filters, kernel_size=conv_1_kernel, strides=conv_1_strides, padding=conv_1_pad, data_format=channel_order, activation=activation_fn, kernel_initializer=kern_init, activity_regularizer=conv_regularizer)(per_image_stand_1)
    spat_1_1 = tf.keras.layers.SpatialDropout2D(rate=spatial_drop_rate_1)(conv_1_1)
    pool_1_1 = tf.keras.layers.MaxPooling2D(pool_size=pool_1_size, padding=pool_1_pad, data_format=channel_order)(spat_1_1)


    conv_2_1 = tf.keras.layers.Conv2D(filters=conv_2_filters, kernel_size=conv_2_kernel, strides=conv_2_strides, padding=conv_2_pad, data_format=channel_order, activation=activation_fn, kernel_initializer=kern_init, activity_regularizer=conv_regularizer)(pool_1_1)
    spat_2_1 = tf.keras.layers.SpatialDropout2D(rate=spatial_drop_rate_2)(conv_2_1)
    pool_2_1 = tf.keras.layers.MaxPooling2D(pool_size=pool_2_size, padding=pool_2_pad, data_format=channel_order)(spat_2_1)


    conv_3_1 = tf.keras.layers.Conv2D(filters=conv_3_filters, kernel_size=conv_3_kernel, strides=conv_3_strides, padding=conv_3_pad, data_format=channel_order, activation=activation_fn, kernel_initializer=kern_init, activity_regularizer=conv_regularizer)(pool_2_1)
    pool_3_1 = tf.keras.layers.MaxPooling2D(pool_size=pool_3_size, padding=pool_3_pad, data_format=channel_order)(conv_3_1)

    flatten_1_1 = tf.keras.layers.Flatten()(pool_3_1)


    dense_1_f_1 = tf.keras.layers.Dense(units=dense_1_f_units, activation=activation_fn, kernel_initializer=kern_init, activity_regularizer=dense_regularizer_1)(flatten_1_1)
    dense_2_f_1 = tf.keras.layers.Dense(units=dense_2_f_units, activation=activation_fn, kernel_initializer=kern_init, activity_regularizer=dense_regularizer_1)(dense_1_f_1)
    dense_3_f_1 = tf.keras.layers.Dense(units=dense_3_f_units, activation=activation_fn, kernel_initializer=kern_init, activity_regularizer=dense_regularizer_1)(dense_2_f_1)


    per_image_stand_2 = tf.keras.layers.Lambda(lambda frame: tf.image.per_image_standardization(frame))(input_fluoro_2)
    conv_1_2 = tf.keras.layers.Conv2D(filters=conv_1_filters, kernel_size=conv_1_kernel, strides=conv_1_strides, padding=conv_1_pad, data_format=channel_order, activation=activation_fn, kernel_initializer=kern_init, activity_regularizer=conv_regularizer)(per_image_stand_2)
    spat_1_2 = tf.keras.layers.SpatialDropout2D(rate=spatial_drop_rate_1)(conv_1_2)
    pool_1_2 = tf.keras.layers.MaxPooling2D(pool_size=pool_1_size, padding=pool_1_pad, data_format=channel_order)(spat_1_2)


    conv_2_2 = tf.keras.layers.Conv2D(filters=conv_2_filters, kernel_size=conv_2_kernel, strides=conv_2_strides, padding=conv_2_pad, data_format=channel_order, activation=activation_fn, kernel_initializer=kern_init, activity_regularizer=conv_regularizer)(pool_1_2)
    spat_2_2 = tf.keras.layers.SpatialDropout2D(rate=spatial_drop_rate_2)(conv_2_2)
    pool_2_2 = tf.keras.layers.MaxPooling2D(pool_size=pool_2_size, padding=pool_2_pad, data_format=channel_order)(spat_2_2)


    conv_3_2 = tf.keras.layers.Conv2D(filters=conv_3_filters, kernel_size=conv_3_kernel, strides=conv_3_strides, padding=conv_3_pad, data_format=channel_order, activation=activation_fn, kernel_initializer=kern_init, activity_regularizer=conv_regularizer)(pool_2_2)
    pool_3_2 = tf.keras.layers.MaxPooling2D(pool_size=pool_3_size, padding=pool_3_pad, data_format=channel_order)(conv_3_2)

    flatten_1_2 = tf.keras.layers.Flatten()(pool_3_2)


    dense_1_f_2 = tf.keras.layers.Dense(units=dense_1_f_units, activation=activation_fn, kernel_initializer=kern_init, activity_regularizer=dense_regularizer_1)(flatten_1_2)
    dense_2_f_2 = tf.keras.layers.Dense(units=dense_2_f_units, activation=activation_fn, kernel_initializer=kern_init, activity_regularizer=dense_regularizer_1)(dense_1_f_2)
    dense_3_f_2 = tf.keras.layers.Dense(units=dense_3_f_units, activation=activation_fn, kernel_initializer=kern_init, activity_regularizer=dense_regularizer_1)(dense_2_f_2)


    dense_1_cali = tf.keras.layers.Dense(units=dense_1_cali_units, activation=activation_fn, kernel_initializer=kern_init, activity_regularizer=dense_regularizer_1)(input_cali)
    dense_2_cali = tf.keras.layers.Dense(units=dense_2_cali_units, activation=activation_fn, kernel_initializer=kern_init, activity_regularizer=dense_regularizer_1)(dense_1_cali)


    dense_0_comb = tf.keras.layers.concatenate([dense_3_v, dense_3_f_1, dense_3_f_2, dense_2_cali])


    dense_1_comb = tf.keras.layers.Dense(units=dense_1_co_units, activation=activation_fn, kernel_initializer=kern_init, activity_regularizer=dense_regularizer_2)(dense_0_comb)
    dense_drop_1 = tf.keras.layers.Dropout(rate=drop_1_comb_rate)(dense_1_comb)
    dense_2_comb = tf.keras.layers.Dense(units=dense_2_co_units, activation=activation_fn, kernel_initializer=kern_init, activity_regularizer=dense_regularizer_2)(dense_drop_1)
    dense_3_comb = tf.keras.layers.Dense(units=dense_3_co_units, activation=activation_fn, kernel_initializer=kern_init, activity_regularizer=dense_regularizer_2)(dense_2_comb)
    dense_4_comb = tf.keras.layers.Dense(units=dense_4_co_units, activation=activation_fn, kernel_initializer=kern_init, activity_regularizer=dense_regularizer_2)(dense_3_comb)


    main_output = tf.keras.layers.Dense(units=main_output_units, activation=main_output_act, kernel_initializer=kern_init, name='main_output')(dense_4_comb)


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


