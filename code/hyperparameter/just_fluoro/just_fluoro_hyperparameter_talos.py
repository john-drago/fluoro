import numpy as np
import h5py
import tensorflow as tf
import os
import sys
import keras
import talos
from sklearn.model_selection import train_test_split
import pickle

save_dir = os.path.abspath(os.path.expanduser('~/fluoro/code/hyperparameter/talos_1'))
os.makedirs(save_dir,exist_ok=True)
expr_name = 'just_fluoro_talos'
expr_no = '1'


def data_comp():

    image_file = h5py.File(os.path.expanduser('~/fluoro/data/compilation/images.h5py'), 'r')
    image_init = image_file['image_dset']
    image_mat = image_init[:]

    label_file = h5py.File(os.path.expanduser('~/fluoro/data/compilation/labels.h5py'), 'r')
    label_init = label_file['labels_dset']
    label_mat = label_init[:]

    cali_file = h5py.File(os.path.expanduser('~/fluoro/data/compilation/calibration.h5py'), 'r')
    cali_init = cali_file['cali_len3_rot']
    cali_mat = cali_init[:]

    image_train_cum, image_test, cali_train_cum, cali_test, label_train_cum, label_test = train_test_split(image_mat,cali_mat,label_mat,shuffle=True,test_size=0.2)

    # print('Image mat size:',image_mat.shape)
    # print('Label mat size:',label_mat.shape)
    # print('Cali mat size:',cali_mat.shape)

    # print('Image cum size:',image_train_cum.shape)
    # print('Label cum size:',label_train_cum.shape)
    # print('Cali cum size:',cali_train_cum.shape)

    # print('Image test size:',image_test.shape)
    # print('Label test size:',label_test.shape)
    # print('Cali test size:',cali_test.shape)


    image_train_sub, image_val, cali_train_sub, cali_val, label_train_sub, label_val = train_test_split(image_train_cum, cali_train_cum, label_train_cum, shuffle=True, test_size=0.2)

    # print('Image sub size:',image_train_sub.shape)
    # print('Label sub size:',label_train_sub.shape)
    # print('Cali sub size:',cali_train_sub.shape)

    # print('Image val size:',image_val.shape)
    # print('Label val size:',label_val.shape)
    # print('Cali val size:',cali_val.shape)

    return image_train_sub, image_val, cali_train_sub, cali_val, label_train_sub, label_val
    # return image_train_cum, cali_train_cum, label_train_cum


params = {
    'regularizer_l1':(0, 1, 10),
    'regularizer_l2':(0, 1, 10),
    'activation_fn':['elu', 'relu'],
    'kern_init':['glorot_uniform', 'glorot_normal'],
    'conv_1_filters':[10, 20, 40, 50],
    'conv_1_kernel':[(10, 10), (5, 5),(3, 3)],
    'conv_1_strides':[(2, 2), (1, 1)],
    'spatial_drop_rate_1':(0, 1, 10),
    'pool_1_size':[(2, 2), (3, 3)],
    'conv_2_filters':[20, 40, 80],
    'conv_2_kernel':[(3, 3), (5, 5)],
    'conv_2_strides':[(2, 2), (1, 1)],
    'pool_2_size':[(2, 2), (3, 3)],
    'conv_3_filters':[20, 80, 100],
    'conv_3_kernel':[(2, 2), (3, 3)],
    'conv_3_strides':[(2, 2), (1, 1)],
    'dense_1_f_units':[40, 80, 120],
    # 'dense_2_f_units':[40, 80, 120],
    # 'dense_3_f_units':[40, 80, 120],
    # 'dense_1_ca_units':[6, 20, 60],
    # 'dense_2_co_units':[20, 40, 80],
    # 'dense_3_co_units':[20, 40, 80],
    # 'drop_1_comb_rate':(0, 1, 10),
    # 'model_opt' :[keras.optimizers.Adam,keras.optimizers.Nadam],
    # 'model_epochs' :[30,40,50,100],
    # 'model_batchsize' :[5,10,30],
    # 'learning_rate' :(0.0001,100,10)
}


# In[11]:


def fluoro_model(X_talos,y_talos,X_val,y_val,params):
    def root_mean_squared_error(y_true, y_pred):
        return keras.backend.sqrt(keras.backend.mean(keras.backend.square(y_pred - y_true)))

    channel_order = 'channels_last'
    img_input_shape = (128,128,1)

    # Hyperparameters
    regularizer = keras.regularizers.l1_l2(l1 = params['regularizer_l1'], l2 = params['regularizer_l2'])
    activation_fn = params['activation_fn']
    kern_init = params['kern_init']

    conv_1_filters = params['conv_1_filters']
    conv_1_kernel = params['conv_1_kernel']
    conv_1_strides = params['conv_1_strides']
    conv_1_padding = 'valid'

    spatial_drop_rate_1 = params['spatial_drop_rate_1']

    pool_1_size = params['pool_1_size']
    pool_1_padding = 'same'

    conv_2_filters = params['conv_2_filters']
    conv_2_kernel = params['conv_2_kernel']
    conv_2_strides = params['conv_2_strides']
    conv_2_padding = 'same'

    pool_2_size = params['pool_2_size']
    pool_2_padding = 'same'

    conv_3_filters = params['conv_3_filters']
    conv_3_kernel = params['conv_3_kernel']
    conv_3_strides = params['conv_3_strides']
    conv_3_padding = 'valid'

    pool_3_size = (2, 2)
    pool_3_padding = 'valid'

    dense_1_f_units = params['dense_1_f_units']
    dense_1_f_bias = True

    # dense_2_f_units = params['dense_2_f_units']
    dense_2_f_units = 120
    dense_2_f_bias = True

    # dense_3_f_units = params['dense_3_f_units']
    dense_3_f_units = 120
    dense_3_f_bias = True

    # dense_1_ca_units = params['dense_1_ca_units']
    dense_1_ca_units = 60
    dense_1_ca_bias = True

    # dense_2_co_units = params['dense_2_co_units']
    dense_2_co_units = 80
    dense_2_co_bias = True

    # drop_1_comb_rate = params['drop_1_comb_rate']
    drop_1_comb_rate = 0.1

    # dense_3_co_units = params['dense_3_co_units']
    dense_3_co_units = 80
    dense_3_co_bias = True

    main_output_units = 6
    main_output_act = 'linear'


    # model_opt = params['model_opt'](lr=params('learning_rate'))
    model_opt = 'adam'
    model_loss = 'mse'
    model_metric = root_mean_squared_error

    # model_epochs = params['model_epochs']
    # model_batchsize = params['model_batchsize']
    model_epochs = 50
    model_batchsize = 10

    input_fluoro_1 = keras.Input(shape=img_input_shape, dtype = 'float32', name='fluoro1_inpt')
    input_fluoro_2 = keras.Input(shape=img_input_shape, dtype = 'float32', name='fluoro2_inpt')
    input_cali = keras.Input(shape=(6,), dtype = 'float32', name = 'cali_inpt')

    bn_1_1 = keras.layers.BatchNormalization()(input_fluoro_1)
    conv_1_1 = keras.layers.Conv2D(filters=conv_1_filters,kernel_size=conv_1_kernel,strides=conv_1_strides,padding=conv_1_padding,activation = activation_fn,input_shape = img_input_shape, data_format = channel_order,activity_regularizer=regularizer,kernel_initializer = kern_init)(bn_1_1)
    spat_1_1 = keras.layers.SpatialDropout2D(rate=spatial_drop_rate_1)(conv_1_1)
    pool_1_1 = keras.layers.MaxPooling2D(pool_size=pool_1_size, padding=pool_1_padding,data_format = channel_order,activity_regularizer=regularizer)(spat_1_1)
    conv_2_1 = keras.layers.Conv2D(filters=conv_2_filters,kernel_size=conv_2_kernel,strides=conv_2_strides,padding=conv_2_padding, activation = activation_fn,data_format = channel_order,activity_regularizer=regularizer,kernel_initializer = kern_init)(pool_1_1)
    pool_2_1 = keras.layers.MaxPooling2D(pool_size=pool_2_size, padding=pool_2_padding,data_format = channel_order,activity_regularizer=regularizer)(conv_2_1)
    conv_3_1 = keras.layers.Conv2D(filters=conv_3_filters,kernel_size=conv_3_kernel,strides=conv_3_strides,padding=conv_3_padding,data_format = channel_order,activity_regularizer=regularizer,kernel_initializer = kern_init)(pool_2_1)
    pool_3_1 = keras.layers.MaxPooling2D(pool_size=pool_3_size,padding=pool_3_padding,data_format = channel_order,activity_regularizer=regularizer)(conv_3_1)
    flatten_1_1 = keras.layers.Flatten()(pool_3_1)
    dense_1_f_1 = keras.layers.Dense(units=dense_1_f_units, activation=activation_fn, use_bias=dense_1_f_bias, kernel_initializer=kern_init, activity_regularizer=regularizer, name = 'dense_1_f_1')(flatten_1_1)
    dense_2_f_1 = keras.layers.Dense(units=dense_2_f_units, activation=activation_fn, use_bias=dense_2_f_bias, kernel_initializer=kern_init, activity_regularizer=regularizer, name = 'dense_2_f_1')(dense_1_f_1)
    dense_3_f_1 = keras.layers.Dense(units=dense_3_f_units, activation=activation_fn, use_bias=dense_3_f_bias, kernel_initializer=kern_init, activity_regularizer=regularizer, name = 'dense_3_f_1')(dense_2_f_1)

    bn_1_2 = keras.layers.BatchNormalization()(input_fluoro_2)
    conv_1_2 = keras.layers.Conv2D(filters=conv_1_filters,kernel_size=conv_1_kernel,strides=conv_1_strides,padding=conv_1_padding,activation = activation_fn,input_shape = img_input_shape, data_format = channel_order,activity_regularizer=regularizer,kernel_initializer = kern_init)(bn_1_2)
    spat_1_2 = keras.layers.SpatialDropout2D(rate=spatial_drop_rate_1)(conv_1_2)
    pool_1_2 = keras.layers.MaxPooling2D(pool_size=pool_1_size, padding=pool_1_padding,data_format = channel_order,activity_regularizer=regularizer)(spat_1_2)
    conv_2_2 = keras.layers.Conv2D(filters=conv_2_filters,kernel_size=conv_2_kernel,strides=conv_2_strides,padding=conv_2_padding, activation = activation_fn,data_format = channel_order,activity_regularizer=regularizer,kernel_initializer = kern_init)(pool_1_2)
    pool_2_2 = keras.layers.MaxPooling2D(pool_size=pool_2_size, padding=pool_2_padding,data_format = channel_order,activity_regularizer=regularizer)(conv_2_2)
    conv_3_2 = keras.layers.Conv2D(filters=conv_3_filters,kernel_size=conv_3_kernel,strides=conv_3_strides,padding=conv_3_padding,data_format = channel_order,activity_regularizer=regularizer,kernel_initializer = kern_init)(pool_2_2)
    pool_3_2 = keras.layers.MaxPooling2D(pool_size=pool_3_size,padding=pool_3_padding,data_format = channel_order,activity_regularizer=regularizer)(conv_3_2)
    flatten_1_2 = keras.layers.Flatten()(pool_3_2)
    dense_1_f_2 = keras.layers.Dense(units=dense_1_f_units, activation=activation_fn, use_bias=dense_1_f_bias, kernel_initializer=kern_init, activity_regularizer=regularizer, name = 'dense_1_f_2')(flatten_1_2)
    dense_2_f_2 = keras.layers.Dense(units=dense_2_f_units, activation=activation_fn, use_bias=dense_2_f_bias, kernel_initializer=kern_init, activity_regularizer=regularizer, name = 'dense_2_f_2')(dense_1_f_2)
    dense_3_f_2 = keras.layers.Dense(units=dense_3_f_units, activation=activation_fn, use_bias=dense_3_f_bias, kernel_initializer=kern_init, activity_regularizer=regularizer, name = 'dense_3_f_2')(dense_2_f_2)

    dense_1_cali =  keras.layers.Dense(units=dense_1_ca_units, activation = activation_fn, use_bias = dense_1_ca_bias, kernel_initializer=kern_init, name = 'dense_1_cali')(input_cali)

    dense_1_comb = keras.layers.concatenate([dense_3_f_1, dense_3_f_2, dense_1_cali], name = 'dense_1_comb')

    dense_2_comb = keras.layers.Dense(units=dense_2_co_units, activation = activation_fn, use_bias = dense_2_co_bias, kernel_initializer=kern_init, name = 'dense_2_comb')(dense_1_comb)
    drop_1_comb = keras.layers.Dropout(rate=drop_1_comb_rate)(dense_2_comb)
    dense_3_comb = keras.layers.Dense(units=dense_3_co_units,activation =activation_fn,use_bias=dense_3_co_bias,kernel_initializer=kern_init, name='dense_3_comb')(drop_1_comb)
    main_output = keras.layers.Dense(units=main_output_units, activation = main_output_act, name = 'main_output')(dense_3_comb)

    model = keras.Model(inputs = [input_fluoro_1,input_fluoro_2,input_cali], outputs = main_output)

    keras.utils.plot_model(model, os.path.abspath(os.path.join(save_dir,expr_name+'_'+expr_no+'.png')), show_shapes=True)

    model.compile(optimizer=model_opt,loss = model_loss, metrics = [model_metric])

    result = model.fit(x=[np.expand_dims(X_talos[0][:,0,:,:],axis=3),np.expand_dims(X_talos[0][:,1,:,:],axis=3), X_talos[1]],y=y_talos,epochs = model_epochs,batch_size = model_batchsize,validation_data=([np.expand_dims(X_val[0][:,0,:,:],axis=3),np.expand_dims(X_val[0][:,1,:,:],axis=3),X_val[1]],y_val), shuffle = True,verbose=1)
    return result, model


image_train_sub, image_val, cali_train_sub, cali_val, label_train_sub, label_val = data_comp()

# fluoro_model([image_train_cum,cali_train_cum],label_train_cum,params)

print('\n'*10)
print('Here we go: ')
print('\n'*10)

t = talos.Scan(x = [image_train_sub,cali_train_sub], y=label_train_sub, x_val = [image_val,cali_val], y_val = label_val, params=params, model=fluoro_model, grid_downsample = 0.5,random_method = 'uniform_mersenne',clear_tf_session=True, print_params=True,dataset_name=expr_name, experiment_no=expr_no,debug=True)

with open(os.path.abspath(os.path.join(save_dir,expr_name+'_'+expr_no+'.pkl')),'wb') as scan_file:
    pickle.dump(t,scan_file,protocol=-1)

print('\n\n\n')
print('-------------')
print('\n')
print('t.data')
print('\n')
print(t.data)
print('\n')
print('-------------')
print('\n\n\n')

print('\n\n\n')
print('-------------')
print('\n')
print('t.details')
print('\n')
print(t.details)
print('\n')
print('-------------')
print('\n\n\n')

print('\n\n\n')
print('-------------')
print('\n')
print('t.saved_models')
print('\n')
print(t.saved_models)
print('\n')
print('-------------')
print('\n\n\n')

print('\n\n\n')
print('-------------')
print('\n')
print('t.saved_weights')
print('\n')
print(t.saved_weights)
print('\n')
print('-------------')
print('\n\n\n')

print('\n\n\n')
print('-------------')
print('\n')
print('t.data')
print('\n')
print(t.data)
print('\n')
print('-------------')
print('\n\n\n')