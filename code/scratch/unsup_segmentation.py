import numpy as np
import h5py
import tensorflow as tf
import keras
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans


expr_name = sys.argv[0][:-3]
expr_no = '2'
save_dir = os.path.abspath(os.path.join(os.path.expanduser('~/fluoro/code/scratch/unsup_seg'), expr_name))
os.makedirs(save_dir, exist_ok=True)


def data_comp(number_of_samples=None):

    # vox_file = h5py.File(os.path.expanduser('~/fluoro/data/compilation/voxels_pad.h5py'), 'r')
    # vox_init = vox_file['vox_dset']
    # vox_mat = vox_init[:number_of_samples]
    # vox_file.close()

    image_file = h5py.File(os.path.expanduser('~/fluoro/data/compilation/images.h5py'), 'r')
    image_init = image_file['image_dset']
    image_mat = image_init[:number_of_samples]
    image_file.close()

    # label_file = h5py.File(os.path.expanduser('~/fluoro/data/compilation/labels.h5py'), 'r')
    # label_init = label_file['labels_dset']
    # label_mat = label_init[:number_of_samples]
    # label_file.close()

    # cali_file = h5py.File(os.path.expanduser('~/fluoro/data/compilation/calibration.h5py'), 'r')
    # cali_init = cali_file['cali_len3_rot']
    # cali_mat = cali_init[:number_of_samples]
    # cali_file.close()

    image_train_cum, image_test = train_test_split(image_mat, shuffle=True, test_size=0.2, random_state=42)

    # print('Image mat size:',image_mat.shape)
    # print('Label mat size:',label_mat.shape)
    # print('Cali mat size:',cali_mat.shape)

    # print('Image cum size:',image_train_cum.shape)
    # print('Label cum size:',label_train_cum.shape)
    # print('Cali cum size:',cali_train_cum.shape)

    # print('Image test size:',image_test.shape)
    # print('Label test size:',label_test.shape)
    # print('Cali test size:',cali_test.shape)


    image_train_sub, image_val = train_test_split(image_train_cum, shuffle=True, test_size=0.2, random_state=42)

    print('Image sub size:', image_train_sub.shape)
    # print('Label sub size:', label_train_sub.shape)
    # print('Cali sub size:', cali_train_sub.shape)

    print('Image val size:', image_val.shape)
    # print('Label val size:', label_val.shape)
    # print('Cali val size:', cali_val.shape)

    # print(vox_mat.shape, image_mat.shape, cali_mat.shape)

    return image_train_sub, image_val
    # return image_train_cum, cali_train_cum, label_train_cum


# -----------------------------------------------------------------


class KMeansLayer(keras.layers.Layer):
    def __init__(self, clusters=8, n_init=5, trainable=False, **kwargs):
        super(KMeansLayer, self).__init__(**kwargs)
        self.clusters = clusters
        self.n_init = n_init

    def build(self, input_shape):
        # self.input_shape = input_shape
        # print(input_shape[0])
        self.output_s = (input_shape[0],input_shape[1], input_shape[2],1)
        self.depth = input_shape[3]
        self.built=True

    def call(self, inputs):
        output=tf.Variable(initial_value=tf.keras.backend.random_uniform(shape=(6,128,128,1)),dtype=tf.float32,trainable=False,validate_shape=False)
        # output=tf.Variable(initial_value=tf.keras.backend.random_uniform(shape=tf.convert_to_tensor(inputs.get_shape()[0],inputs.get_shape()[1],inputs.get_shape()[2],1)),dtype=tf.float32)
        def KMeansFunc(input_tens,clusters=self.clusters,n_init=self.n_init):
            base_mat = np.zeros((input_tens.shape[0],input_tens.shape[1],input_tens.shape[2]))
            for frame in range(input_tens.shape[0]):
                init_mat = np.zeros((input_tens.shape[1]*input_tens.shape[2]))
                print(init_mat.shape)
                reshape_mat = np.reshape(input_tens[frame],(input_tens.shape[1]*input_tens.shape[2],input_tens.shape[3]))
                print(reshape_mat.shape)
                kmeans_init = KMeans(n_clusters=clusters, n_init=n_init)
                class_pred = kmeans_init.fit_predict(reshape_mat)
                for clust in range(8):
                    init_mat[class_pred==clust] = np.mean(reshape_mat[class_pred==clust],axis=1)
                    init_mat[class_pred==clust] = np.mean(reshape_mat[class_pred==clust],None)
                print(base_mat.shape)
                base_mat[frame]=np.reshape(init_mat,(input_tens.shape[1],input_tens.shape[2]))
            return np.expand_dims(base_mat,axis=-1).astype('float32')
        output = tf.py_func(KMeansFunc,[inputs],tf.float32) + self.kernel-self.kernel
        # output=tf.placeholder(shape=(inputs.get_shape()[0],inputs.get_shape()[1],inputs.get_shape()[2],1),dtype=tf.float32)
        return output

        


    def compute_output_shape(self, input_shape):
        return self.output_s

# -----------------------------------------------------------------

channel_order = 'channels_last'
img_input_shape = (128, 128, 1)
# vox_input_shape = (198, 162, 564, 1)
# cali_input_shape = (6,)


def root_mean_squared_error(y_true, y_pred):
    return keras.backend.sqrt(keras.backend.mean(keras.backend.square(y_pred - y_true)))


params = {

    # 2D CONV
    'conv_1_1_filters': 64,
    'conv_1_1_kernel': 5,
    'conv_1_1_strides': 1,
    'conv_1_1_pad': 'same',
    'spatial_drop_rate_1_1': 0.5,
    'conv_1_2_filters': 64,
    'conv_1_2_kernel': 3,
    'conv_1_2_strides': 1,
    'conv_1_2_pad': 'same',
    'spatial_drop_rate_1_2': 0.5,
    'pool_1_size': 2,
    'pool_1_pad': 'same',
    'conv_2_1_filters': 128,
    'conv_2_1_kernel': 3,
    'conv_2_1_strides': 1,
    'conv_2_1_pad': 'same',
    'spatial_drop_rate_2_1': 0.5,
    'conv_2_2_filters': 128,
    'conv_2_2_kernel': 3,
    'conv_2_2_strides': 1,
    'conv_2_2_pad': 'same',
    'spatial_drop_rate_2_2': 0.5,
    'pool_2_size': 2,
    'pool_2_pad': 'same',
    'conv_3_1_filters': 256,
    'conv_3_1_kernel': 3,
    'conv_3_1_strides': 1,
    'conv_3_1_pad': 'same',
    'spatial_drop_rate_3_1': 0.5,
    'conv_3_2_filters': 256,
    'conv_3_2_kernel': 3,
    'conv_3_2_strides': 1,
    'conv_3_2_pad': 'same',
    'spatial_drop_rate_3_2': 0.5,
    'pool_3_size': 2,
    'pool_3_pad': 'same',
    'conv_4_1_filters': 512,
    'conv_4_1_kernel': 3,
    'conv_4_1_strides': 1,
    'conv_4_1_pad': 'same',
    'spatial_drop_rate_4_1': 0.5,
    'conv_4_2_filters': 512,
    'conv_4_2_kernel': 3,
    'conv_4_2_strides': 1,
    'conv_4_2_pad': 'same',
    'spatial_drop_rate_4_2': 0.5,
    'pool_4_size': 2,
    'pool_4_pad': 'same',
    'conv_5_1_filters': 1024,
    'conv_5_1_kernel': 3,
    'conv_5_1_strides': 1,
    'conv_5_1_pad': 'same',
    'conv_5_2_filters': 1024,
    'conv_5_2_kernel': 3,
    'conv_5_2_strides': 1,
    'conv_5_2_pad': 'same',
    'up_conv_1_filters': 512,
    'up_conv_1_kernel': 2,
    'up_conv_1_strides': 1,
    'up_conv_1_pad': 'same',
    'up_1_size': 2,
    'up_1_int': 'bilinear',
    'conv_6_1_filters': 512,
    'conv_6_1_kernel': 3,
    'conv_6_1_strides': 1,
    'conv_6_1_pad': 'same',
    'conv_6_2_filters': 512,
    'conv_6_2_kernel': 3,
    'conv_6_2_strides': 1,
    'conv_6_2_pad': 'same',
    'up_conv_2_filters': 256,
    'up_conv_2_kernel': 2,
    'up_conv_2_strides': 1,
    'up_conv_2_pad': 'same',
    'up_2_size': 2,
    'up_2_int': 'bilinear',
    'conv_7_1_filters': 256,
    'conv_7_1_kernel': 3,
    'conv_7_1_strides': 1,
    'conv_7_1_pad': 'same',
    'conv_7_2_filters': 256,
    'conv_7_2_kernel': 3,
    'conv_7_2_strides': 1,
    'conv_7_2_pad': 'same',
    'up_conv_3_filters': 128,
    'up_conv_3_kernel': 2,
    'up_conv_3_strides': 1,
    'up_conv_3_pad': 'same',
    'up_3_size': 2,
    'up_3_int': 'bilinear',
    'conv_8_1_filters': 128,
    'conv_8_1_kernel': 3,
    'conv_8_1_strides': 1,
    'conv_8_1_pad': 'same',
    'conv_8_2_filters': 128,
    'conv_8_2_kernel': 3,
    'conv_8_2_strides': 1,
    'conv_8_2_pad': 'same',
    'up_conv_4_filters': 64,
    'up_conv_4_kernel': 2,
    'up_conv_4_strides': 1,
    'up_conv_4_pad': 'same',
    'up_4_size': 2,
    'up_4_int': 'bilinear',
    'conv_9_1_filters': 64,
    'conv_9_1_kernel': 3,
    'conv_9_1_strides': 1,
    'conv_9_1_pad': 'same',
    'conv_9_2_filters': 64,
    'conv_9_2_kernel': 64,
    'conv_9_2_strides': 1,
    'conv_9_2_pad': 'same',
    'conv_k_1_filters': 20,
    'conv_k_1_kernel': 3,
    'conv_k_1_strides': 1,
    'conv_k_1_pad': 'same',
    'conv_k_2_filters': 3,
    'conv_k_2_kernel': 1,
    'conv_k_2_strides': 1,
    'conv_k_2_pad': 'same',


    # General Housekeeping
    'regularizer_l1': 0.1,
    'regularizer_l2': 0.25,
    'activation_fn': 'elu',
    'kern_init': 'glorot_uniform',
    'model_opt': keras.optimizers.RMSprop(),
    'learning_rate': 0.001,
    'model_epochs': 50,
    'model_batchsize': 6,
    'model_loss': 'mse',
    'model_metric': 'mse'

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

# input_vox = keras.Input(shape=vox_input_shape, name='input_vox', tensor=vox_ph)
# input_fluoro_1 = keras.Input(shape=img_input_shape, name='input_fluoro_1', tensor=fluoro_1_ph)
# input_fluoro_2 = keras.Input(shape=img_input_shape, name='input_fluoro_2', tensor=fluoro_2_ph)
# input_cali = keras.Input(shape=cali_input_shape, name='input_cali', tensor=cali_ph)

# -----------------------------------------------------------------

# Input Layers
input_fluoro_1 = keras.Input(shape=img_input_shape, name='input_fluoro_1', dtype='float32')

# -----------------------------------------------------------------

# d
bn_1 = keras.layers.BatchNormalization(input_shape=img_input_shape)(input_fluoro_1)
conv_1_1 = keras.layers.Conv2D(filters=params['conv_1_1_filters'], kernel_size=params['conv_1_1_kernel'], strides=params['conv_1_1_strides'], padding=params['conv_1_1_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'])(bn_1)
spat_1_1 = keras.layers.SpatialDropout2D(rate=params['spatial_drop_rate_1_1'])(conv_1_1)
conv_1_2 = keras.layers.Conv2D(filters=params['conv_1_2_filters'], kernel_size=params['conv_1_2_kernel'], strides=params['conv_1_2_strides'], padding=params['conv_1_2_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'])(spat_1_1)
spat_1_2 = keras.layers.SpatialDropout2D(rate=params['spatial_drop_rate_1_2'])(conv_1_2)
pool_1 = keras.layers.MaxPooling2D(pool_size=params['pool_1_size'], padding=params['pool_1_pad'], data_format=channel_order)(spat_1_2)

conv_2_1 = keras.layers.SeparableConv2D(filters=params['conv_2_1_filters'], kernel_size=params['conv_2_1_kernel'], strides=params['conv_2_1_strides'], padding=params['conv_2_1_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'])(pool_1)
spat_2_1 = keras.layers.SpatialDropout2D(rate=params['spatial_drop_rate_2_1'])(conv_2_1)
conv_2_2 = keras.layers.SeparableConv2D(filters=params['conv_2_2_filters'], kernel_size=params['conv_2_2_kernel'], strides=params['conv_2_2_strides'], padding=params['conv_2_2_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'])(spat_2_1)
spat_2_2 = keras.layers.SpatialDropout2D(rate=params['spatial_drop_rate_2_2'])(conv_2_2)
pool_2 = keras.layers.MaxPooling2D(pool_size=params['pool_2_size'], padding=params['pool_2_pad'], data_format=channel_order)(spat_2_2)

conv_3_1 = keras.layers.SeparableConv2D(filters=params['conv_3_1_filters'], kernel_size=params['conv_3_1_kernel'], strides=params['conv_3_1_strides'], padding=params['conv_3_1_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'])(pool_2)
spat_3_1 = keras.layers.SpatialDropout2D(rate=params['spatial_drop_rate_3_1'])(conv_3_1)
conv_3_2 = keras.layers.SeparableConv2D(filters=params['conv_3_2_filters'], kernel_size=params['conv_3_2_kernel'], strides=params['conv_3_2_strides'], padding=params['conv_3_2_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'])(spat_3_1)
spat_3_2 = keras.layers.SpatialDropout2D(rate=params['spatial_drop_rate_3_2'])(conv_3_2)
pool_3 = keras.layers.MaxPooling2D(pool_size=params['pool_3_size'], padding=params['pool_3_pad'], data_format=channel_order)(spat_3_2)

conv_4_1 = keras.layers.SeparableConv2D(filters=params['conv_4_1_filters'], kernel_size=params['conv_4_1_kernel'], strides=params['conv_4_1_strides'], padding=params['conv_4_1_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'])(pool_3)
spat_4_1 = keras.layers.SpatialDropout2D(rate=params['spatial_drop_rate_4_1'])(conv_4_1)
conv_4_2 = keras.layers.SeparableConv2D(filters=params['conv_4_2_filters'], kernel_size=params['conv_4_2_kernel'], strides=params['conv_4_2_strides'], padding=params['conv_4_2_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'])(spat_4_1)
spat_4_2 = keras.layers.SpatialDropout2D(rate=params['spatial_drop_rate_4_2'])(conv_4_2)
pool_4 = keras.layers.MaxPooling2D(pool_size=params['pool_4_size'], padding=params['pool_4_pad'], data_format=channel_order)(spat_4_2)

conv_5_1 = keras.layers.SeparableConv2D(filters=params['conv_5_1_filters'], kernel_size=params['conv_5_1_kernel'], strides=params['conv_5_1_strides'], padding=params['conv_5_1_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'])(pool_4)
conv_5_2 = keras.layers.SeparableConv2D(filters=params['conv_5_2_filters'], kernel_size=params['conv_5_2_kernel'], strides=params['conv_5_2_strides'], padding=params['conv_5_2_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'])(conv_5_1)

up_conv_1 = keras.layers.SeparableConv2D(filters=params['up_conv_1_filters'], kernel_size=params['up_conv_1_kernel'], strides=params['up_conv_1_strides'], padding=params['up_conv_1_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'])(conv_5_2)
up_1 = keras.layers.UpSampling2D(size=(params['up_1_size'], params['up_1_size']), interpolation=params['up_1_int'])(up_conv_1)
conv_6_1 = keras.layers.SeparableConv2D(filters=params['conv_6_1_filters'], kernel_size=params['conv_6_1_kernel'], strides=params['conv_6_1_strides'], padding=params['conv_6_1_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'])(up_1)
conv_6_2 = keras.layers.SeparableConv2D(filters=params['conv_6_2_filters'], kernel_size=params['conv_6_2_kernel'], strides=params['conv_6_2_strides'], padding=params['conv_6_2_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'])(conv_6_1)

up_conv_2 = keras.layers.SeparableConv2D(filters=params['up_conv_2_filters'], kernel_size=params['up_conv_2_kernel'], strides=params['up_conv_2_strides'], padding=params['up_conv_2_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'])(conv_6_2)
up_2 = keras.layers.UpSampling2D(size=(params['up_2_size'], params['up_2_size']), interpolation=params['up_2_int'])(up_conv_2)
conv_7_1 = keras.layers.SeparableConv2D(filters=params['conv_7_1_filters'], kernel_size=params['conv_7_1_kernel'], strides=params['conv_7_1_strides'], padding=params['conv_7_1_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'])(up_2)
conv_7_2 = keras.layers.SeparableConv2D(filters=params['conv_7_2_filters'], kernel_size=params['conv_7_2_kernel'], strides=params['conv_7_2_strides'], padding=params['conv_7_2_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'])(conv_7_1)

up_conv_3 = keras.layers.SeparableConv2D(filters=params['up_conv_3_filters'], kernel_size=params['up_conv_3_kernel'], strides=params['up_conv_3_strides'], padding=params['up_conv_3_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'])(conv_7_2)
up_3 = keras.layers.UpSampling2D(size=(params['up_3_size'], params['up_3_size']), interpolation=params['up_3_int'])(up_conv_3)
conv_8_1 = keras.layers.SeparableConv2D(filters=params['conv_8_1_filters'], kernel_size=params['conv_8_1_kernel'], strides=params['conv_8_1_strides'], padding=params['conv_8_1_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'])(up_3)
conv_8_2 = keras.layers.SeparableConv2D(filters=params['conv_8_2_filters'], kernel_size=params['conv_8_2_kernel'], strides=params['conv_8_2_strides'], padding=params['conv_8_2_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'])(conv_8_1)

up_conv_4 = keras.layers.SeparableConv2D(filters=params['up_conv_4_filters'], kernel_size=params['up_conv_4_kernel'], strides=params['up_conv_2_strides'], padding=params['up_conv_4_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'])(conv_8_2)
up_4 = keras.layers.UpSampling2D(size=(params['up_4_size'], params['up_4_size']), interpolation=params['up_4_int'])(up_conv_4)
conv_9_1 = keras.layers.SeparableConv2D(filters=params['conv_9_1_filters'], kernel_size=params['conv_9_1_kernel'], strides=params['conv_9_1_strides'], padding=params['conv_9_1_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'])(up_4)
conv_9_2 = keras.layers.SeparableConv2D(filters=params['conv_9_2_filters'], kernel_size=params['conv_9_2_kernel'], strides=params['conv_9_2_strides'], padding=params['conv_9_2_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'])(conv_9_1)

conv_k_1 = keras.layers.SeparableConv2D(filters=params['conv_k_1_filters'], kernel_size=params['conv_k_1_kernel'], strides=params['conv_k_1_strides'], padding=params['conv_k_1_pad'], data_format=channel_order, activation=params['activation_fn'], kernel_initializer=params['kern_init'])(conv_9_2)

# kmeans_out = KMeansLayer(clusters=8,n_init=5)(conv_k_1)
# kmeans_out.trainable=False

conv_k_2 = keras.layers.SeparableConv2D(filters=params['conv_k_2_filters'], kernel_size=params['conv_k_2_kernel'], strides=params['conv_k_2_strides'], padding=params['conv_k_2_pad'], data_format=channel_order, activation='linear', kernel_initializer=params['kern_init'])(conv_k_1)




kmeans_out = keras.layers.Lambda(function=KMeansFunc)(conv_k_2)

# kmeans_out = keras.layers.SeparableConv2D(filters=1, kernel_size=1, strides=1, padding='same', data_format=channel_order, activation='linear', kernel_initializer=params['kern_init'], use_bias=False)(conv_k_2)




# def KMeansFunc(x):
#     # batch_mat = np.zeros((x.shape[0],x.shape[1],x.shape[2]))
#     def inner_fn(x, clusters=8,n_init=5):
#         batch_mat = np.zeros((x.shape[0],x.shape[1],x.shape[2]))
#         for frame in range(keras.backend.shape(x)[0]):
#             input_mat = x[frame]

#             init_mat = np.zeros((input_mat.shape[0]*input_mat.shape[1]))
#             kmeans_init = KMeans(n_clusters=clusters,n_init=n_init)
#             reshape_mat = np.reshape(input_mat,(input_mat.shape[0]*input_mat.shape[1],input_mat.shape[2]))
#             class_pred = kmeans_init.fit_predict(reshape_mat)

#             for clust in range(clusters):
#                 init_mat[class_pred==clust] = np.mean(reshape_mat[class_pred==clust],axis=1)
#                 init_mat[class_pred==clust] = np.mean(reshape_mat[class_pred==clust],None)

#             batch_mat[frame] = np.reshape(init_mat,(input_mat.shape[0],input_mat.shape[1])).astype(np.float32)
#         batch_mat = np.expand_dims(batch_mat,axis=-1)
#         return batch_mat.astype(np.float32)
#     return tf.py_func(inner_fn,[x],tf.float32)

# def KMeansFunc_outputshape(input_shape):
#     return (input_shape[0],input_shape[1], input_shape[2],1)

# kmeans_out = keras.layers.Lambda(KMeansFunc)(conv_k_2)

# print(dir(kmeans_out))
# print(kmeans_out.graph)


# -----------------------------------------------------------------

# Main Output
# main_output = keras.layers.Dense(units=params['main_output_units'], activation=params['main_output_act'], kernel_initializer=params['kern_init'], name='main_output')(conv_k_1)

        
# -----------------------------------------------------------------


# Model Housekeeping
model = keras.Model(inputs=[input_fluoro_1], outputs=kmeans_out)
keras.utils.plot_model(model, os.path.abspath(os.path.join(save_dir, expr_name + '_' + expr_no + '.png')), show_shapes=True)
model.compile(optimizer=params['model_opt'], loss=params['model_loss'], metrics=[params['model_metric']])

# image_train_sub, image_val = data_comp(200)

# result = model.fit(x={'input_vox': np.expand_dims(vox_train_sub, axis=-1), 'input_fluoro_1': np.expand_dims(image_train_sub[:, 0, :, :], axis=-1), 'input_fluoro_2': np.expand_dims(image_train_sub[:, 1, :, :], axis=-1), 'input_cali': cali_train_sub}, y=label_train_sub, validation_data=([np.expand_dims(vox_val, axis=-1), np.expand_dims(image_val[:, 0, :, :], axis=-1), np.expand_dims(image_val[:, 1, :, :], axis=-1), cali_val], label_val), epochs=params['model_epochs'], batch_size=params['model_batchsize'], shuffle=True, verbose=2)

# model.save(os.path.join(os.getcwd(), 'test_model_save.h5'))







# def KMeansFunc(input_tens,clusters=8,n_init=5):
#     base_mat = np.zeros((1,128,128,1))
#     global xaaa
#     xaaa = 0
#     def KMeans_base(input_tens,base_mat=base_mat):
#         global xaaa
#         xaaa +=1
#         init_mat = np.zeros((input_tens.shape[0]*input_tens.shape[1]))
#         print(init_mat.shape)
#         reshape_mat = np.reshape(input_tens[frame],(input_tens.shape[0]*input_tens.shape[1],input_tens.shape[2]))
#         print(reshape_mat.shape)
#         kmeans_init = KMeans(n_clusters=clusters, n_init=n_init)
#         class_pred = kmeans_init.fit_predict(reshape_mat)
#         for clust in range(8):
#             init_mat[class_pred==clust] = np.mean(reshape_mat[class_pred==clust],axis=1)
#             init_mat[class_pred==clust] = np.mean(reshape_mat[class_pred==clust],None)
#         print(base_mat.shape)
#         base_mat[frame]=np.reshape(init_mat,(input_tens.shape[1],input_tens.shape[2]))
#         return np.expand_dims(base_mat,axis=-1).astype('float32')











# class KMeansLayer(keras.layers.Layer):
#     def __init__(self, clusters, n_init, trainable=False, **kwargs):
#         super(KMeansLayer, self).__init__(**kwargs)
#         self.clusters = clusters
#         self.n_init = n_init

#     def build(self, input_shape):
#         # self.input_shape = input_shape
#         input_shape = input_shape
#         self.output_s = (input_shape[0],input_shape[1], input_shape[2],1)
#         self.depth = input_shape[3]
#         super(KMeansLayer, self).build(input_shape)

#     def call(self, inputs, **kwargs):

#         def KMeansFunc(input_tens):
#             batch_mat = np.zeros((input_tens.shape[0],input_tens.shape[1],input_tens.shape[2]))
#             for frame in range(input_tens.shape[0]):
#                 input_mat = input_tens[frame]

#                 init_mat = np.zeros((input_mat.shape[0]*input_mat.shape[1]))
#                 kmeans_init = KMeans(n_clusters=self.clusters,n_init=self.n_init)
#                 reshape_mat = np.reshape(input_mat,(input_mat.shape[0]*input_mat.shape[1],input_mat.shape[2]))
#                 class_pred = kmeans_init.fit_predict(reshape_mat)

#                 for clust in range(clusters):
#                     init_mat[class_pred==clust] = np.mean(reshape_mat[class_pred==clust],axis=1)
#                     init_mat[class_pred==clust] = np.mean(reshape_mat[class_pred==clust],None)

#                 batch_mat[frame] = np.reshape(init_mat,(input_mat.shape[0],input_mat.shape[1])).astype(np.float32)
#             batch_mat = np.expand_dims(batch_mat,axis=-1)
#             return tf.convert_to_tensor(batch_mat.astype(np.float32))



#         return tf.py_func(KMeansFunc,[inputs],tf.float32)


#     def compute_output_shape(self, input_shape):
#         return self.output_s