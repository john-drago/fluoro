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
    # 3D CONV
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


    'v_pool_0_size': 2,
    'v_pool_0_pad': 'same',


    'v_conv_2_filters': 30,
    'v_conv_2_kernel': 5,
    'v_conv_2_strides_0': 1,
    'v_conv_2_strides_1': 1,
    'v_conv_2_strides_2': 1,
    'v_conv_2_pad': 'same',

    'v_spatial_drop_rate_1': 0.3,

    'v_conv_3_filters': 30,
    'v_conv_3_kernel': 5,
    'v_conv_3_strides_0': 1,
    'v_conv_3_strides_1': 1,
    'v_conv_3_strides_2': 1,
    'v_conv_3_pad': 'same',


    'v_pool_1_size': 2,
    'v_pool_1_pad': 'same',


    'v_pool_2_size': 2,
    'v_pool_2_pad': 'same',


    'v_conv_4_filters': 30,
    'v_conv_4_kernel': 5,
    'v_conv_4_strides_0': 2,
    'v_conv_4_strides_1': 2,
    'v_conv_4_strides_2': 2,
    'v_conv_4_pad': 'same',

    'v_conv_5_filters': 30,
    'v_conv_5_kernel': 5,
    'v_conv_5_strides_0': 1,
    'v_conv_5_strides_1': 1,
    'v_conv_5_strides_2': 1,
    'v_conv_5_pad': 'same',

    'v_spatial_drop_rate_2': 0.3,

    'v_conv_6_filters': 30,
    'v_conv_6_kernel': 1,
    'v_conv_6_strides_0': 2,
    'v_conv_6_strides_1': 2,
    'v_conv_6_strides_2': 2,
    'v_conv_6_pad': 'same',






    'dense_1_v_units': 300,
    'dense_2_v_units': 250,
    'dense_3_v_units': 250,
    'dense_4_v_units': 200,


    # 2D CONV
    'conv_1_filters': 30,
    'conv_1_kernel': 5,
    'conv_1_strides': 2,
    'conv_1_pad': 'same',
    'spatial_drop_rate_1': 0.3,

    'conv_2_filters': 40,
    'conv_2_kernel': 3,
    'conv_2_strides': 2,
    'conv_2_pad': 'same',
    'pool_1_size': 2,
    'pool_1_pad': 'same',

    'conv_3_filters': 50,
    'conv_3_kernel': 3,
    'conv_3_strides': 2,
    'conv_3_pad': 'same',
    'spatial_drop_rate_2': 0.3,

    'conv_4_filters': 60,
    'conv_4_kernel': 3,
    'conv_4_strides': 2,
    'conv_4_pad': 'same',
    'pool_2_size': 2,
    'pool_2_pad': 'same',

    'conv_5_filters': 60,
    'conv_5_kernel': 3,
    'conv_5_strides': 1,
    'conv_5_pad': 'same',
    'spatial_drop_rate_3': 0.3,

    'conv_6_filters': 30,
    'conv_6_kernel': 3,
    'conv_6_strides': 1,
    'conv_6_pad': 'same',

    'dense_1_f_units': 120,
    'dense_2_f_units': 120,
    'dense_3_f_units': 80,

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


v_conv_0 = tf.keras.layers.Conv3D(filters=params['v_conv_0_filters'], kernel_size=params['v_conv_0_kernel'], strides=(params['v_conv_0_strides_0'], params['v_conv_0_strides_1'], params['v_conv_0_strides_2']), padding=params['v_conv_0_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['v_conv_regularizer'])(input_vox)
bn_0 = tf.keras.layers.BatchNormalization()(v_conv_0)


v_spat_0 = tf.keras.layers.SpatialDropout3D(rate=params['v_spatial_drop_rate_0'])(bn_0)
v_conv_1 = tf.keras.layers.Conv3D(filters=params['v_conv_1_filters'], kernel_size=params['v_conv_1_kernel'], strides=(params['v_conv_1_strides_0'], params['v_conv_1_strides_1'], params['v_conv_1_strides_2']), padding=params['v_conv_1_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['v_conv_regularizer'])(v_spat_0)


v_pool_0 = tf.keras.layers.MaxPooling3D(pool_size=params['v_pool_0_size'], padding=params['v_pool_0_pad'], data_format=channel_order)(v_conv_1)


bn_1 = tf.keras.layers.BatchNormalization()(v_pool_0)
v_conv_2 = tf.keras.layers.Conv3D(filters=params['v_conv_2_filters'], kernel_size=params['v_conv_2_kernel'], strides=(params['v_conv_2_strides_0'], params['v_conv_2_strides_1'], params['v_conv_2_strides_2']), padding=params['v_conv_2_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['v_conv_regularizer'])(bn_1)
bn_2 = tf.keras.layers.BatchNormalization()(v_conv_2)
v_spat_1 = tf.keras.layers.SpatialDropout3D(rate=params['v_spatial_drop_rate_1'])(bn_2)
v_conv_3 = tf.keras.layers.Conv3D(filters=params['v_conv_3_filters'], kernel_size=params['v_conv_3_kernel'], strides=(params['v_conv_3_strides_0'], params['v_conv_3_strides_1'], params['v_conv_3_strides_2']), padding=params['v_conv_3_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['v_conv_regularizer'])(v_spat_1)
v_add_0 = tf.keras.layers.Add()([v_conv_3, v_conv_2])


v_pool_1 = tf.keras.layers.MaxPooling3D(pool_size=params['v_pool_1_size'], padding=params['v_pool_1_pad'], data_format=channel_order)(v_add_0)


bn_1 = tf.keras.layers.BatchNormalization()(v_pool_1)
v_conv_2 = tf.keras.layers.Conv3D(filters=params['v_conv_2_filters'], kernel_size=params['v_conv_2_kernel'], strides=(params['v_conv_2_strides_0'], params['v_conv_2_strides_1'], params['v_conv_2_strides_2']), padding=params['v_conv_2_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['v_conv_regularizer'])(bn_1)
bn_2 = tf.keras.layers.BatchNormalization()(v_conv_2)
v_spat_1 = tf.keras.layers.SpatialDropout3D(rate=params['v_spatial_drop_rate_1'])(bn_2)
v_conv_3 = tf.keras.layers.Conv3D(filters=params['v_conv_3_filters'], kernel_size=params['v_conv_3_kernel'], strides=(params['v_conv_3_strides_0'], params['v_conv_3_strides_1'], params['v_conv_3_strides_2']), padding=params['v_conv_3_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['v_conv_regularizer'])(v_spat_1)
v_add_0 = tf.keras.layers.Add()([v_conv_3, bn_1])


bn_3 = tf.keras.layers.BatchNormalization()(v_add_0)
v_conv_4 = tf.keras.layers.Conv3D(filters=params['v_conv_4_filters'], kernel_size=params['v_conv_4_kernel'], strides=(params['v_conv_4_strides_0'], params['v_conv_4_strides_1'], params['v_conv_4_strides_2']), padding=params['v_conv_4_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['v_conv_regularizer'])(bn_3)
bn_4 = tf.keras.layers.BatchNormalization()(v_conv_4)
v_spat_2 = tf.keras.layers.SpatialDropout3D(rate=params['v_spatial_drop_rate_2'])(bn_4)
v_conv_5 = tf.keras.layers.Conv3D(filters=params['v_conv_5_filters'], kernel_size=params['v_conv_5_kernel'], strides=(params['v_conv_5_strides_0'], params['v_conv_5_strides_1'], params['v_conv_5_strides_2']), padding=params['v_conv_5_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['v_conv_regularizer'])(v_spat_2)
v_conv_6 = tf.keras.layers.Conv3D(filters=params['v_conv_6_filters'], kernel_size=params['v_conv_6_kernel'], strides=(params['v_conv_6_strides_0'], params['v_conv_6_strides_1'], params['v_conv_6_strides_2']), padding=params['v_conv_6_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['v_conv_regularizer'])(bn_3)
v_add_0 = tf.keras.layers.Add()([v_conv_5, v_conv_6])


v_pool_2 = tf.keras.layers.MaxPooling3D(pool_size=params['v_pool_2_size'], padding=params['v_pool_2_pad'], data_format=channel_order)(v_add_0)


bn_1 = tf.keras.layers.BatchNormalization()(v_pool_2)
v_conv_2 = tf.keras.layers.Conv3D(filters=params['v_conv_2_filters'], kernel_size=params['v_conv_2_kernel'], strides=(params['v_conv_2_strides_0'], params['v_conv_2_strides_1'], params['v_conv_2_strides_2']), padding=params['v_conv_2_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['v_conv_regularizer'])(bn_1)
bn_2 = tf.keras.layers.BatchNormalization()(v_conv_2)
v_spat_1 = tf.keras.layers.SpatialDropout3D(rate=params['v_spatial_drop_rate_1'])(bn_2)
v_conv_3 = tf.keras.layers.Conv3D(filters=params['v_conv_2_filters'], kernel_size=params['v_conv_2_kernel'], strides=(params['v_conv_2_strides_0'], params['v_conv_2_strides_1'], params['v_conv_2_strides_2']), padding=params['v_conv_2_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['v_conv_regularizer'])(v_spat_1)
v_add_0 = tf.keras.layers.Add()([v_conv_3, bn_1])


v_flatten_0 = tf.keras.layers.Flatten()(v_add_0)



# v_spat_1 = tf.keras.layers.SpatialDropout3D(rate=params['v_spatial_drop_rate_1'])(v_conv_1)

# v_conv_2 = tf.keras.layers.Conv3D(filters=params['v_conv_2_filters'], kernel_size=params['v_conv_2_kernel'], strides=(params['v_conv_2_strides_0'], params['v_conv_2_strides_1'], params['v_conv_2_strides_2']), padding=params['v_conv_2_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['v_conv_regularizer'])(v_spat_1)
# v_pool_1 = tf.keras.layers.MaxPooling3D(pool_size=params['v_pool_1_size'], padding=params['v_pool_1_pad'], data_format=channel_order)(v_conv_2)

# v_conv_3 = tf.keras.layers.Conv3D(filters=params['v_conv_3_filters'], kernel_size=params['v_conv_3_kernel'], strides=(params['v_conv_3_strides_0'], params['v_conv_3_strides_1'], params['v_conv_3_strides_2']), padding=params['v_conv_3_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['v_conv_regularizer'])(v_pool_1)
# v_spat_2 = tf.keras.layers.SpatialDropout3D(rate=params['v_spatial_drop_rate_2'])(v_conv_3)


# v_conv_4 = tf.keras.layers.Conv3D(filters=params['v_conv_4_filters'], kernel_size=params['v_conv_4_kernel'], strides=(params['v_conv_4_strides_0'], params['v_conv_4_strides_1'], params['v_conv_4_strides_2']), padding=params['v_conv_4_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['v_conv_regularizer'])(v_spat_2)
# v_pool_2 = tf.keras.layers.MaxPooling3D(pool_size=params['v_pool_2_size'], padding=params['v_pool_2_pad'], data_format=channel_order)(v_conv_4)

# v_conv_5 = tf.keras.layers.Conv3D(filters=params['v_conv_5_filters'], kernel_size=params['v_conv_5_kernel'], strides=(params['v_conv_5_strides_0'], params['v_conv_5_strides_1'], params['v_conv_5_strides_2']), padding=params['v_conv_5_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['v_conv_regularizer'])(v_conv_4)
# v_spat_3 = tf.keras.layers.SpatialDropout3D(rate=params['v_spatial_drop_rate_3'])(v_conv_5)

# v_conv_6 = tf.keras.layers.Conv3D(filters=params['v_conv_6_filters'], kernel_size=params['v_conv_6_kernel'], strides=(params['v_conv_6_strides_0'], params['v_conv_6_strides_1'], params['v_conv_6_strides_2']), padding=params['v_conv_6_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['v_conv_regularizer'])(v_spat_3)

# v_conv_7 = tf.keras.layers.Conv3D(filters=params['v_conv_7_filters'], kernel_size=params['v_conv_7_kernel'], strides=(params['v_conv_7_strides_0'], params['v_conv_7_strides_1'], params['v_conv_7_strides_2']), padding=params['v_conv_7_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['v_conv_regularizer'])(v_conv_6)
# v_spat_4 = tf.keras.layers.SpatialDropout3D(rate=params['v_spatial_drop_rate_4'])(v_conv_7)

# v_conv_8 = tf.keras.layers.Conv3D(filters=params['v_conv_8_filters'], kernel_size=params['v_conv_8_kernel'], strides=(params['v_conv_8_strides_0'], params['v_conv_8_strides_1'], params['v_conv_8_strides_2']), padding=params['v_conv_8_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['v_conv_regularizer'])(v_conv_7)


# v_flatten_1 = tf.keras.layers.Flatten()(v_conv_8)


dense_1_v = tf.keras.layers.Dense(units=params['dense_1_v_units'], activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['dense_regularizer_1'])(v_flatten_0)
dense_2_v = tf.keras.layers.Dense(units=params['dense_2_v_units'], activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['dense_regularizer_1'])(dense_1_v)
dense_3_v = tf.keras.layers.Dense(units=params['dense_3_v_units'], activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['dense_regularizer_1'])(dense_2_v)
dense_4_v = tf.keras.layers.Dense(units=params['dense_4_v_units'], activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['dense_regularizer_1'])(dense_3_v)

# -----------------------------------------------------------------

# per_image_stand_1 = tf.keras.layers.Lambda(lambda frame: tf.image.per_image_standardization(frame))(input_fluoro_1)

# -----------------------------------------------------------------

# per_image_stand_2 = tf.keras.layers.Lambda(lambda frame: tf.image.per_image_standardization(frame))(input_fluoro_2)

# conv_1_2 = tf.keras.layers.Conv2D(filters=params['conv_1_filters'], kernel_size=params['conv_1_kernel'], strides=params['conv_1_strides'], padding=params['conv_1_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'], activity_regularizer=params['conv_regularizer'])(per_image_stand_2)
# pool_2_2 = tf.keras.layers.MaxPooling2D(pool_size=params['pool_2_size'], padding=params['pool_2_pad'], data_format=channel_order)(conv_4_2)



# -----------------------------------------------------------------



# -----------------------------------------------------------------



# -----------------------------------------------------------------



# -----------------------------------------------------------------

# Main Output
main_output = tf.keras.layers.Dense(units=params['main_output_units'], activation=params['main_output_act'], kernel_initializer=params['kern_init'], name='main_output')(dense_1_v)

# -----------------------------------------------------------------

# Model Housekeeping
# model = tf.keras.Model(inputs=[input_vox, input_fluoro_1, input_fluoro_2, input_cali], outputs=main_output)
model = tf.keras.Model(inputs=[input_vox], outputs=main_output)

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





