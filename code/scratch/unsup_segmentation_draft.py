import numpy as np
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
import keras

import sys


expr_name = sys.argv[0][:-3]
expr_no = '1'
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

channel_order = 'channels_last'
input_size = (128, 128, 1)
# vox_input_shape = (198, 162, 564, 1)
# cali_input_shape = (6,)


inputs = Input(input_size)
conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
drop4 = Dropout(0.5)(conv4)
pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
drop5 = Dropout(0.5)(conv5)

up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(drop5))
merge6 = concatenate([drop4, up6], axis=3)
conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv6))
merge7 = concatenate([conv3, up7], axis=3)
conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv7))
merge8 = concatenate([conv2, up8], axis=3)
conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv8))
merge9 = concatenate([conv1, up9], axis=3)
conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

model = Model(input=inputs, output=conv10)

model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

# model.summary()
keras.utils.plot_model(model, os.path.abspath(os.path.join(save_dir, expr_name + '_' + expr_no + '.png')), show_shapes=True)
# model.compile(optimizer=params['model_opt'], loss=params['model_loss'], metrics=[params['model_metric']])

# vox_train_sub, vox_val, image_train_sub, image_val, cali_train_sub, cali_val, label_train_sub, label_val = data_comp(2000)

# result = model.fit(x={'input_vox': np.expand_dims(vox_train_sub, axis=-1), 'input_fluoro_1': np.expand_dims(image_train_sub[:, 0, :, :], axis=-1), 'input_fluoro_2': np.expand_dims(image_train_sub[:, 1, :, :], axis=-1), 'input_cali': cali_train_sub}, y=label_train_sub, validation_data=([np.expand_dims(vox_val, axis=-1), np.expand_dims(image_val[:, 0, :, :], axis=-1), np.expand_dims(image_val[:, 1, :, :], axis=-1), cali_val], label_val), epochs=params['model_epochs'], batch_size=params['model_batchsize'], shuffle=True, verbose=2)

# model.save(os.path.join(os.getcwd(), 'test_model_save.h5'))
