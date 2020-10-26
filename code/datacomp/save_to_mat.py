'''
This module will devise some functions that can create a .mat file in the expected format of the .mat files that will determine the position and rotation of the bone for each frame.
'''

import numpy as np
import scipy.io as sio
import os
from coord_change import Angles2Basis



def generate_save_mat(dict_data, filename):
    '''
    We are expecting an input of a dictionary with
    'Tibia_R'
    'Tibia_V'
    'Femur_R'
    'Femur_V'

    Femur is cup
    Tibia is stem
    '''

    dt = [('R', 'O'), ('V', 'O')]

    Cup_RV = np.zeros((1,), dtype=dt)
    Cup_RV[0]['R'] = dict_data['Femur_R']
    Cup_RV[0]['V'] = dict_data['Femur_V']

    Stem_RV = np.zeros((1,), dtype=dt)
    Stem_RV[0]['R'] = dict_data['Tibia_R']
    Stem_RV[0]['V'] = dict_data['Tibia_V']

    sio.savemat(filename, {'Cup_RV': Cup_RV, 'Stem_RV': Stem_RV})

    return None


def extract_base_filename(path_to_frame):
    list_of_frames = sorted(os.listdir(path_to_frame))
    while True:
        frame_guess = np.random.choice(list_of_frames)
        if (frame_guess != '.DS_Store') and (frame_guess != 'cali'):
            list_of_save_files = os.listdir(os.path.join(path_to_frame, frame_guess))

            for save_file in list_of_save_files:
                if save_file[-11:] == 'results.mat':
                    return save_file[:-16]


def create_output_vars_from_predicted(output_mat):

    rots = output_mat[:, 0:3]
    trans = output_mat[:, 3:6]

    rot_output = np.zeros((rots.shape[0], 3, 3))
    trans_ouput = trans

    ticker = 0
    for instance in rots:
        rot_output[ticker, :, :] = Angles2Basis(instance)
        ticker += 1


    return rot_output, trans_ouput


def save_new_mats_from_training(model_predict_data, list_of_frames_matched, prediction_save_dir_path=os.path.expanduser('~/fluoro/data/prediction')):
    '''
    Assumes we made an evaluation matrix from a dataset, where we know what each entry in the matrix corresponds to. There should be an accompanying list of path to frames, where each entry corresponds to the instance evaluated along the first dimension by the model.


    This function will take a matrix with the predicted values along with the accompanying list_of_frames_matched. It will then generate corresponding mat files for each frame predicted with both bones loaded into the mat file.
    '''


    if len(model_predict_data) != len(list_of_frames_matched * 2):
        raise ValueError('The length of model_predict_data and list_of_frames_matched need to be equivalent.')

    ticker = 0
    for frame_path in list_of_frames_matched:

        save_sub_path = ''
        for dir1 in frame_path.split(os.sep)[-4:-1]:
            save_sub_path = os.path.join(save_sub_path, dir1)
        frame = frame_path.split(os.sep)[-1]


        save_path = os.path.join(prediction_save_dir_path, save_sub_path)
        os.makedirs(save_path, exist_ok=True)

        rot_output, trans_output = create_output_vars_from_predicted(model_predict_data[2 * ticker: 2 * ticker + 2])

        output_dict = {}
        output_dict['Femur_R'] = rot_output[0]
        output_dict['Femur_V'] = trans_output[0]
        output_dict['Tibia_R'] = rot_output[1]
        output_dict['Tibia_V'] = trans_output[1]

        base_file_name = extract_base_filename(os.path.abspath(frame_path[:-4]))
        mat_filename = base_file_name + frame + '_results.mat'

        print(os.path.join(save_path, mat_filename))


        generate_save_mat(dict_data=output_dict, filename=os.path.join(save_path, mat_filename))


        ticker += 1

    return None







if __name__ == '__main__':
    import tensorflow as tf
    import pickle

    # -----------------------------------------------------------------

# This code will test whether or not the above code can create new .mat files from a simulated data set containing simulated predictions of rotation and translation

    # dir_file = open(os.path.join(os.getcwd(), 'vox_fluoro_hist_objects.pkl'), 'rb')

    # dict_of_paths = pickle.load(dir_file)

    # dict_of_frames = dict_of_paths['dict_of_frames']
    # list_of_path_to_frames = dict_of_paths['list_of_path_to_frames']

    # random_activity_and_patient = np.random.choice(list_of_path_to_frames)

    # int_list_of_test_frames = dict_of_frames[random_activity_and_patient]
    # list_of_predict_frames = []

    # for frame in int_list_of_test_frames:
    #     list_of_predict_frames.append(os.path.abspath(os.path.join(random_activity_and_patient, frame)))

    # list_of_predict_frames.sort()

    # test_dataset = np.random.rand(2 * len(list_of_predict_frames), 6)

    # save_new_mats_from_training(test_dataset, list_of_predict_frames)

    # dir_file.close()



    # -----------------------------------------------------------------

    # tf.keras.models.load_model()

    # -----------------------------------------------------------------

    # test_dict = {}
    # test_dict['Femur_R'] = np.random.rand(3, 3)
    # test_dict['Femur_V'] = np.random.rand(3)
    # test_dict['Tibia_R'] = np.random.rand(3, 3)
    # test_dict['Tibia_V'] = np.random.rand(3)

    # save_path = os.path.join(os.getcwd(), 'test_1.mat')



