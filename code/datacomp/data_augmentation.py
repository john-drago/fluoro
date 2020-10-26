'''
This function will perform data augmentation on our current data set. Basically, we will do small translations and rotations on our voxel dataset to increase the number of instances we are currently training with.
'''

import os
import scipy.io as sio
import skimage
import numpy as np
import trimesh
import pandas as pd
import h5py
import pickle
import psutil
import datetime


# ---------------------------------------------------------------
# Before we get started, need to first define some top-level variables, which future functions will make reference to.

top_level_dir = os.path.expanduser('~/fluoro/data')
save_dir = os.path.abspath('/Volumes/Seagate/fluoro')

# ---------------------------------------------------------------
# These are some functions, which will be useful for changing the data for the augmentation.


def Global2Local_Coord(rot_mat, trans_vector, points_in_global):
    '''
    func Global2Local_Coord(rot_mat, trans_vector, points_in_global)

    - Takes "rotation matrix", whereby the columns form an orthonormal basis, describing the axes of the new coordinate system in terms of the global coordinate system: Should be of form 3x3. Matrix should be square and invertible.
    [ e_1  e_2  e_3 ]

    - Takes translation vector of size 3, which describes translation from global origin to new local origin (global origin ----> local origin).

    - Takes points defined in the global coordinate frame.

    - Returns positions (which were originally defined in the global coordinate frame) in new local coordinate frame.
    '''
    if rot_mat.shape[0] != rot_mat.shape[1]:
        raise ValueError('Rotation Matrix should be square')
    # elif trans_vector.shape != (3,) and trans_vector.shape != (1, 3):
    #     raise ValueError('Translation Matrix should be an array of size 3 or 1x3 matrix')

    translated_points = points_in_global - trans_vector

    points_in_local = np.transpose(np.matmul(np.linalg.inv(rot_mat), np.transpose(translated_points)))

    return points_in_local


def Local2Global_Coord(rot_mat, trans_vector, points_in_local):
    '''
    function Local2Global_Coord(rot_mat, trans_vector, points_in_local)

    - Takes "rotation matrix", whereby the columns form an orthonormal basis. The "rotation matrix" should describe the axes of the new coordinate system in terms of the global coordinate system. The matrix should be 3x3 and be invertible.
    [ e_1  e_2  e_3 ]

    - Takes translation vector of size 3, which describes translation from global origin to the new local origin (global origin ----> local origin).

    - Takes points defined in the local coordinate frame.

    - Returns positions (which were originally defined in the local coordinate frame) in the global coordinate frame.
    '''
    if rot_mat.shape[0] != rot_mat.shape[1]:
        raise ValueError('Rotation Matrix should be square')
    elif trans_vector.shape != (3,) and trans_vector.shape != (1, 3):
        raise ValueError('Translation Matrix should be an array of size 3 or 1x3 matrix')
    # print(rot_mat.shape)
    # print(trans_vector.shape)



    rotated_points = np.transpose(np.matmul(rot_mat, np.transpose(points_in_local)))

    points_in_global = rotated_points + trans_vector

    return points_in_global


def Basis2Angles(rot_mat):
    '''
    function Basis2Angles(rot_mat)

    This function will take a "rotation matrix", whereby the columns form an orthonormal basis. The "rotation matrix" should describe the axes of the new coordinate system in terms of the global coordinate system. Matrix should be 3x3 and invertible.
    [ e_1  e_2  e_3 ]

    We are making the assumption that this rotation matrix is equivalent to three basis transformation in the follow order:

    R_rot = R_z * R_y * R_x (order matters)

    Returns a vector of size 3, which containes the following angles in order:
    - theta, as part of rotation matrix:
             [    1          0              0    ]
    R_x =    [    0     cos(theta)   -sin(theta) ]
             [    0     sin(theta)    cos(theta) ]
    - phi
             [  cos(phi)       0       sin(phi) ]
    R_y =    [    0            1           0    ]
             [ -sin(phi)       0       cos(phi) ]
    - psi
              [  cos(psi)    -sin(psi)      0 ]
    R_z =     [  sin(psi)     cos(psi)      0 ]
              [     0            0          1 ]
    '''


    phi = np.arcsin(-rot_mat[2, 0])
    psi = np.arcsin((rot_mat[1, 0]) / (np.cos(phi)))
    if rot_mat[0, 0] / np.cos(phi) < 0:
        psi = np.pi - psi


    theta = np.arcsin((rot_mat[2, 1]) / (np.cos(phi)))
    if rot_mat[2, 2] / np.cos(phi) < 0:
        theta = np.pi - theta

    rot_mat_guess = Angles2Basis([theta, phi, psi])

    error = rot_mat_guess - rot_mat


    epsilon = 0.000009


    error_binary = (error < epsilon)

    if not error_binary.all():
        phi = np.pi - phi
        psi = np.arcsin((rot_mat[1, 0]) / (np.cos(phi)))
        if rot_mat[0, 0] / np.cos(phi) < 0:
            psi = np.pi - psi

        theta = np.arcsin((rot_mat[2, 1]) / (np.cos(phi)))
        # if rot_mat[2, 2] / np.cos(phi) < 0:
        #     theta = np.pi - theta

        rot_mat_guess = Angles2Basis(np.array(theta, phi, psi))
        error = rot_mat_guess - rot_mat
        epsilon = 0.000009
        error_binary = (error < epsilon)





    assert error_binary.all()
    return np.array([theta, phi, psi])


def u_x(angle_array):
    '''
    Computes corresponding part of rotation matrix, seen below:

             [   U_x     V_x     W_x   ]
    R_rot =  [   U_y     V_y     W_y   ]
             [   U_z     V_z     W_z   ]

    With the assumption that R_rot = R_z * R_y * R_x (order matters)

             [    1          0              0    ]
    R_x =    [    0     cos(theta)   -sin(theta) ]
             [    0     sin(theta)    cos(theta) ]

             [  cos(phi)       0       sin(phi) ]
    R_y =    [    0       cos(theta)       0    ]
             [ -sin(phi)       0       cos(phi) ]

              [  cos(psi)  -sin(psi)    0 ]
    R_z =     [  sin(phi)   cos(psi)    0 ]
              [     0          0        1 ]

    '''
    return np.cos(angle_array[1]) * np.cos(angle_array[2])


def u_y(angle_array):
    '''
    Computes corresponding part of rotation matrix, seen below:

             [   U_x     V_x     W_x   ]
    R_rot =  [   U_y     V_y     W_y   ]
             [   U_z     V_z     W_z   ]

    With the assumption that R_rot = R_z * R_y * R_x (order matters)

             [    1          0              0    ]
    R_x =    [    0     cos(theta)   -sin(theta) ]
             [    0     sin(theta)    cos(theta) ]

             [  cos(phi)       0       sin(phi) ]
    R_y =    [    0       cos(theta)       0    ]
             [ -sin(phi)       0       cos(phi) ]

              [  cos(psi)  -sin(psi)    0 ]
    R_z =     [  sin(phi)   cos(psi)    0 ]
              [     0          0        1 ]

    '''
    return np.cos(angle_array[1]) * np.sin(angle_array[2])


def u_z(angle_array):
    '''
    Computes corresponding part of rotation matrix, seen below:

             [   U_x     V_x     W_x   ]
    R_rot =  [   U_y     V_y     W_y   ]
             [   U_z     V_z     W_z   ]

    With the assumption that R_rot = R_z * R_y * R_x (order matters)

             [    1          0              0    ]
    R_x =    [    0     cos(theta)   -sin(theta) ]
             [    0     sin(theta)    cos(theta) ]

             [  cos(phi)       0       sin(phi) ]
    R_y =    [    0       cos(theta)       0    ]
             [ -sin(phi)       0       cos(phi) ]

              [  cos(psi)  -sin(psi)    0 ]
    R_z =     [  sin(phi)   cos(psi)    0 ]
              [     0          0        1 ]

    '''
    return -np.sin(angle_array[1])


def v_x(angle_array):
    '''
    Computes corresponding part of rotation matrix, seen below:

             [   U_x     V_x     W_x   ]
    R_rot =  [   U_y     V_y     W_y   ]
             [   U_z     V_z     W_z   ]

    With the assumption that R_rot = R_z * R_y * R_x (order matters)

             [    1          0              0    ]
    R_x =    [    0     cos(theta)   -sin(theta) ]
             [    0     sin(theta)    cos(theta) ]

             [  cos(phi)       0       sin(phi) ]
    R_y =    [    0       cos(theta)       0    ]
             [ -sin(phi)       0       cos(phi) ]

              [  cos(psi)  -sin(psi)    0 ]
    R_z =     [  sin(phi)   cos(psi)    0 ]
              [     0          0        1 ]

    '''
    return np.cos(angle_array[2]) * np.sin(angle_array[0]) * np.sin(angle_array[1]) - np.cos(angle_array[0]) * np.sin(angle_array[2])


def v_y(angle_array):
    '''
    Computes corresponding part of rotation matrix, seen below:

             [   U_x     V_x     W_x   ]
    R_rot =  [   U_y     V_y     W_y   ]
             [   U_z     V_z     W_z   ]

    With the assumption that R_rot = R_z * R_y * R_x (order matters)

             [    1          0              0    ]
    R_x =    [    0     cos(theta)   -sin(theta) ]
             [    0     sin(theta)    cos(theta) ]

             [  cos(phi)       0       sin(phi) ]
    R_y =    [    0       cos(theta)       0    ]
             [ -sin(phi)       0       cos(phi) ]

              [  cos(psi)  -sin(psi)    0 ]
    R_z =     [  sin(phi)   cos(psi)    0 ]
              [     0          0        1 ]

    '''
    return np.cos(angle_array[0]) * np.cos(angle_array[2]) + np.sin(angle_array[0]) * np.sin(angle_array[1]) * np.sin(angle_array[2])


def v_z(angle_array):
    '''
    Computes corresponding part of rotation matrix, seen below:

             [   U_x     V_x     W_x   ]
    R_rot =  [   U_y     V_y     W_y   ]
             [   U_z     V_z     W_z   ]

    With the assumption that R_rot = R_z * R_y * R_x (order matters)

             [    1          0              0    ]
    R_x =    [    0     cos(theta)   -sin(theta) ]
             [    0     sin(theta)    cos(theta) ]

             [  cos(phi)       0       sin(phi) ]
    R_y =    [    0       cos(theta)       0    ]
             [ -sin(phi)       0       cos(phi) ]

              [  cos(psi)  -sin(psi)    0 ]
    R_z =     [  sin(phi)   cos(psi)    0 ]
              [     0          0        1 ]

    '''
    return np.cos(angle_array[1]) * np.sin(angle_array[0])


def w_x(angle_array):
    '''
    Computes corresponding part of rotation matrix, seen below:

             [   U_x     V_x     W_x   ]
    R_rot =  [   U_y     V_y     W_y   ]
             [   U_z     V_z     W_z   ]

    With the assumption that R_rot = R_z * R_y * R_x (order matters)

             [    1          0              0    ]
    R_x =    [    0     cos(theta)   -sin(theta) ]
             [    0     sin(theta)    cos(theta) ]

             [  cos(phi)       0       sin(phi) ]
    R_y =    [    0       cos(theta)       0    ]
             [ -sin(phi)       0       cos(phi) ]

              [  cos(psi)  -sin(psi)    0 ]
    R_z =     [  sin(phi)   cos(psi)    0 ]
              [     0          0        1 ]

    '''
    return np.cos(angle_array[0]) * np.cos(angle_array[2]) * np.sin(angle_array[1]) + np.sin(angle_array[0]) * np.sin(angle_array[2])


def w_y(angle_array):
    '''
    Computes corresponding part of rotation matrix, seen below:

             [   U_x     V_x     W_x   ]
    R_rot =  [   U_y     V_y     W_y   ]
             [   U_z     V_z     W_z   ]

    With the assumption that R_rot = R_z * R_y * R_x (order matters)

             [    1          0              0    ]
    R_x =    [    0     cos(theta)   -sin(theta) ]
             [    0     sin(theta)    cos(theta) ]

             [  cos(phi)       0       sin(phi) ]
    R_y =    [    0       cos(theta)       0    ]
             [ -sin(phi)       0       cos(phi) ]

              [  cos(psi)  -sin(psi)    0 ]
    R_z =     [  sin(phi)   cos(psi)    0 ]
              [     0          0        1 ]

    '''
    return np.cos(angle_array[0]) * np.sin(angle_array[1]) * np.sin(angle_array[2]) - np.cos(angle_array[2]) * np.sin(angle_array[0])


def w_z(angle_array):
    '''
    Computes corresponding part of rotation matrix, seen below:

             [   U_x     V_x     W_x   ]
    R_rot =  [   U_y     V_y     W_y   ]
             [   U_z     V_z     W_z   ]

    With the assumption that R_rot = R_z * R_y * R_x (order matters)

             [    1          0              0    ]
    R_x =    [    0     cos(theta)   -sin(theta) ]
             [    0     sin(theta)    cos(theta) ]

             [  cos(phi)       0       sin(phi) ]
    R_y =    [    0       cos(theta)       0    ]
             [ -sin(phi)       0       cos(phi) ]

              [  cos(psi)  -sin(psi)    0 ]
    R_z =     [  sin(phi)   cos(psi)    0 ]
              [     0          0        1 ]

    '''
    return np.cos(angle_array[0]) * np.cos(angle_array[1])


def Angles2Basis(rot_ang_array):
    '''
    function Angles2Basis([theta,phi,psi])

    With these angles, this function will compute the orthonormal basis for the coordinate system rotation according to to the following transformations:

             [    1          0              0    ]
    R_x =    [    0     cos(theta)   -sin(theta) ]
             [    0     sin(theta)    cos(theta) ]

             [  cos(phi)       0       sin(phi) ]
    R_y =    [    0       cos(theta)       0    ]
             [ -sin(phi)       0       cos(phi) ]

              [  cos(psi)  -sin(psi)    0 ]
    R_z =     [  sin(phi)   cos(psi)    0 ]
              [     0          0        1 ]


    We will find the rotation matrix after applying the following rotations, in order:

    R_rot = R_z * R_y * R_x (order matters)

    We will produce a rotation matrix of the form:

             [   U_x     V_x     W_x   ]
    R_rot =  [   U_y     V_y     W_y   ]
             [   U_z     V_z     W_z   ]

    '''
    theta = rot_ang_array[0]
    phi = rot_ang_array[1]
    psi = rot_ang_array[2]

    u_x = np.cos(phi) * np.cos(psi)
    u_y = np.cos(phi) * np.sin(psi)
    u_z = -np.sin(phi)

    v_x = np.cos(psi) * np.sin(theta) * np.sin(phi) - np.cos(theta) * np.sin(psi)
    v_y = np.cos(theta) * np.cos(psi) + np.sin(theta) * np.sin(phi) * np.sin(psi)
    v_z = np.cos(phi) * np.sin(theta)

    w_x = np.cos(theta) * np.cos(psi) * np.sin(phi) + np.sin(theta) * np.sin(psi)
    w_y = np.cos(theta) * np.sin(phi) * np.sin(psi) - np.cos(psi) * np.sin(theta)
    w_z = np.cos(theta) * np.cos(phi)

    rot_mat = np.array([
        [u_x, v_x, w_x],
        [u_y, v_y, w_y],
        [u_z, v_z, w_z]
    ])

    return np.squeeze(rot_mat)


# ---------------------------------------------------------------
# We are first going to copy some old useful functions into this file to make them available for later calls. These functions will primarily focus on identifying where the data is located.


def generate_dict_of_acts_with_patients():
    '''
    This function will generate a dictionary of the different activities and all of the patients who did that activity.

    The final format will be a dictionary where keys are the activities and the values are the different patients who did the activity.

    Assuming that we are in path: */fluoro/data
    '''

    activity_list = []
    pt_dict = {}

    for direct1 in os.listdir(os.path.abspath(top_level_dir)):
        if (direct1 != '.DS_Store') and (direct1 != 'compilation') and (direct1 != 'prediction'):
            activity_list.append(direct1)

            # print(direct1)

            list_of_pts_for_act = []
            for direct2 in os.listdir(os.path.join(top_level_dir, direct1)):


                if direct2 != '.DS_Store':
                    list_of_pts_for_act.append(direct2)
            # print(list_of_pts_for_act)
            pt_dict[direct1] = list_of_pts_for_act
            # print(pt_dict)
    return pt_dict



def generate_dict_path_to_frames(dict_of_act_pts):
    '''
    This function will take a dictionary, where keys are activities and the values are the list of different patients who completed that given activity

    This function will generate a dictionary with keys of path to frames and values of the list of frames at the corresponding path

    Keys: path to directory where frames are
    Values: list of frames
    '''
    path_to_frames_dict = {}

    for act in dict_of_act_pts.keys():
        pt_list = dict_of_act_pts[act]
        for pt in pt_list:
            dir_to_pt = os.path.join(top_level_dir, act, pt)
            for direct3 in os.listdir(dir_to_pt):
                list_of_frames = []
                if direct3 != '.DS_Store' and direct3 != 'stl':
                    dir_to_frames = os.path.join(top_level_dir, act, pt, direct3)
                    # print(dir_to_frames)
                    for frame in os.listdir(dir_to_frames):
                        if (frame != '.DS_Store') and (frame != 'cali'):
                            list_of_frames.append(frame)
                    path_to_frames_dict[dir_to_frames] = list_of_frames
    return path_to_frames_dict


def generate_comprehensive_list_of_frames(dict_path_to_frames):
    '''
    This function will take in the dictionary with the keys as paths to locations of frames and the values as the strings of the frames for a given path.

    It will return a comprehensive list of all of the paths to the various frames that are stored under "~/fluoro/data".
    '''

    list_of_path_to_frames = []

    for frme_path in dict_path_to_frames.keys():
        for frame in dict_path_to_frames[frme_path]:
            list_of_path_to_frames.append(os.path.join(frme_path, frame))

    return sorted(list_of_path_to_frames)

# ---------------------------------------------------------------


def extract_calibration_data(path_to_cali):
    '''
    This function will return the R12 and V12 variables from the reg2fl***.mat file in the "data/activity/patient/laterality/cali" folder.

    extract_calibration_data(path_to_cali)

        input: expects a path to the directory where the cali frame is located

        output: will output an array of the form: [ R12, V12 ]

    '''
    cali_str = 'cali'
    for fle in os.listdir(os.path.join(path_to_cali, cali_str)):
        if fle[0:6].lower() == 'reg2fl':
            fluoro_file = sio.loadmat(os.path.join(path_to_cali, cali_str, fle))
            # print(fle)

            break

    return [fluoro_file['R12'], fluoro_file['V12']]


def extract_image_data(path_to_frame, resize_shape=(128, 128)):
    '''
    This function will return the image data for a given frame from the two fluoroscopes that comprise the viewing area.

    Additionally, the function will reshape the image size from standard 1024 x 1024 input to what is in the 'resize_shape' input

    function: extract_image_data(path_to_frame, resize_shape=(128, 128))

        input:
            path_to_frame: where the frame that contains the image is located

            resize_shape: what the resized image should be

        output: will output an array of the form: [ image1, image 2 ]
    '''
    image_array = [0, 0]

    for fle in os.listdir(os.path.normpath(path_to_frame)):
        if fle[-4:] == '.png' and fle[0:2] == 'F1':
            image_load = skimage.io.imread(os.path.join(path_to_frame, fle))
            image_resize = skimage.transform.resize(image_load, resize_shape, anti_aliasing=True)
            image_gray = skimage.color.rgb2gray(image_resize)
            # image_array[0] = image_gray.reshape(resize_shape[0] * resize_shape[1])
            image_array[0] = image_gray
            # print(type(image_gray))
        elif fle[-4:] == '.png' and fle[0:2] == 'F2':
            image_load = skimage.io.imread(os.path.join(path_to_frame, fle))
            image_resize = skimage.transform.resize(image_load, resize_shape, anti_aliasing=True)
            image_gray = skimage.color.rgb2gray(image_resize)
            # print(type(image_gray))
            # image_array[1] = image_gray.reshape(resize_shape[0] * resize_shape[1])
            image_array[1] = image_gray

    # pass
    return np.array(image_array)


def extract_labels_rot_trans_femur_tib_data(path_to_frame):
    '''
    This function will take in the path to the frame where the data for each registration has occurred. It will return the rotation matrix (converted to three angles) and translation vector for both the FEMUR and the TIBIA.

    function extract_femur_tib_cup_data(path_to_frames)

        input:
            path to frames

        output:
            2x2 matrix:
                [ [ R_angles of femur     V_trans of femur ],
                  [ R_angles of tibia     V_trans of tibia ] ]
    '''
    femur_keyword = 'Cup_RV'
    tibia_keyword = 'Stem_RV'

    # femur_tib_data = [[0, 0], [0, 0]]
    femur_tib_data = np.zeros((2, 6))

    for fle in os.listdir(os.path.normpath(path_to_frame)):
        if fle[-4:] == '.mat':
            results_file = sio.loadmat(os.path.join(path_to_frame, fle))

            femur_data = results_file[femur_keyword]
            tibia_data = results_file[tibia_keyword]

            femur_rot = np.array(Basis2Angles(femur_data[0][0][0]))
            femur_trans = femur_data[0][0][1]

            femur_tib_data[0, 0:3] = femur_rot.reshape(3)
            femur_tib_data[0, 3:6] = femur_trans.reshape(3)

            tibia_rot = np.array(Basis2Angles(tibia_data[0][0][0]))
            tibia_trans = tibia_data[0][0][1]

            femur_tib_data[1, 0:3] = tibia_rot.reshape(3)
            femur_tib_data[1, 3:6] = tibia_trans.reshape(3)

            break

    return femur_tib_data



def voxel_from_array(mesh_vertices, spacing=0.5, mark_origin=False, location_of_origin=np.array([0, 0, 0]), origin_value=2):
    '''
    This function will take in a matrix of the location of mesh vertices. It will then take the vertices and transform them into a binary voxel data set with a 1 located in the bin if a corresponding point is to be found. It will return the voxelized matrix.

    input:
        mesh_vertices --> expects np.array of locations of mesh vertices
        spacing --> the spacing of the voxels in mm
    output:
        bin_mat --> a binary voxelized matrix wtih 1's corresponding to points with a corresponding vertex
    '''
    mesh_min_vec = np.min(mesh_vertices, axis=0)
    mesh_min_vec = np.where(mesh_min_vec > 0, 0, mesh_min_vec)
    mesh_max_vec = np.max(mesh_vertices, axis=0)
    mesh_max_vec = np.where(mesh_max_vec < 0, 0, mesh_max_vec)
    mesh_min_mat = mesh_vertices - mesh_min_vec
    range_vec = mesh_max_vec - mesh_min_vec
    # print('range_vec:\t', range_vec)
    bins_vec = np.ceil(range_vec / spacing)
    # print('bins_vec:\t', bins_vec)
    bin_mat = np.zeros(bins_vec.astype('int32'))
    # print('bin_mat.shape:\t', bin_mat.shape)

    for indx in range(mesh_vertices.shape[0]):
        # print(int(np.floor(mesh_min_mat[indx, 0] / spacing)))
        # print(int(np.floor(mesh_min_mat[indx, 1] / spacing)))
        # print(int(np.floor(mesh_min_mat[indx, 2] / spacing)))

        # print(type(int(np.floor(mesh_min_mat[indx, 0] / spacing))))
        # print(type(int(np.floor(mesh_min_mat[indx, 1] / spacing))))
        # print(type(int(np.floor(mesh_min_mat[indx, 2] / spacing))))


        bin_mat[int(np.floor(mesh_min_mat[indx, 0] / spacing)):int(np.ceil(mesh_min_mat[indx, 0] / spacing)) + 1, int(np.floor(mesh_min_mat[indx, 1] / spacing)):int(np.ceil(mesh_min_mat[indx, 1] / spacing)) + 1, int(np.floor(mesh_min_mat[indx, 2] / spacing)):int(np.ceil(mesh_min_mat[indx, 2] / spacing)) + 1] = 1

    if mark_origin:
        location_of_origin = location_of_origin - mesh_min_vec
        bin_mat[int(np.floor(location_of_origin[0] / spacing)), int(np.floor(location_of_origin[1] / spacing)), int(np.floor(location_of_origin[2] / spacing))] = origin_value

    return bin_mat.astype('int8')



def extract_stl_to_meshpoints(mesh_obj, PTS_file, voxelize_dim=0.5, random_disp=False, random_seed=np.random.randint(low=0, high=2**32)):
    '''

    '''
    # print('Random_seed: ', random_seed)
    np.random.seed(random_seed)

    PTS_file = np.array(PTS_file)


    X_vec = np.array(PTS_file[0, :] - PTS_file[1, :])
    Z_vec_pre = np.array(PTS_file[2, :] - PTS_file[3, :])

    Y_vec = np.cross(Z_vec_pre, X_vec)

    # We have to do the second cross, because we cannot a priori guarantee that the X and Z unit vectors are orthogonal. Once we generate an orthogonal Y unit vector, we can regenerate the Z unit vector based on the X and Y unit vectors to create a true orthonormal basis

    Z_unit = np.cross(X_vec, Y_vec)

    x_unit = X_vec / np.linalg.norm(X_vec)
    y_unit = Y_vec / np.linalg.norm(Y_vec)
    z_unit = Z_unit / np.linalg.norm(Z_unit)


    rot_mat = np.array([
        [x_unit[0], y_unit[0], z_unit[0]],
        [x_unit[1], y_unit[1], z_unit[1]],
        [x_unit[2], y_unit[2], z_unit[2]]])

    origin_mesh_local = np.array(PTS_file[0:2, :].mean(axis=0))

    verts_local_coord = Global2Local_Coord(rot_mat, origin_mesh_local, mesh_obj.vertices)

    if random_disp:
        return random_rotation_translation(verts_local_coord, rotation=True)
    else:
        return verts_local_coord

# ---------------------------------------------------------------
# These are the two main functions, which will be used for data augmentation.


def random_rotation_translation(mesh_vertices, rotation=True, translation=False):
    '''
    This function will take in array of the mesh vertices. It will then apply a random rotation and/or translation to the mesh_vertices. This function will output a new list array of the mesh vertices, which have been rotated and/or translated.

    input:
        mesh_vertices --> the input array of mesh vertices, which should be in their local coordinate frame, such that the local origin is [0,0,0].
        rotation --> whether or not to apply a rotation to the dataset
        translation --> whether or not to apply a random translation to the dataset

    output:
        updated_mesh_vertices --> the updated mesh vertices, which have been randomly transformed
    '''
    if rotation:
        random_theta = np.random.rand(1) * 2 * np.pi
        random_phi = np.random.rand(1) * 2 * np.pi
        random_psi = np.random.rand(1) * 2 * np.pi
    else:
        random_theta = 0
        random_phi = 0
        random_psi = 0

    rotation_angles = np.array([random_theta, random_phi, random_psi])

    if translation:
        random_x = np.random.rand(1) * 3
        random_y = np.random.rand(1) * 3
        random_z = np.random.rand(1) * 3
    else:
        random_x = 0
        random_y = 0
        random_z = 0
    translation_vecs = np.array([random_x, random_y, random_z])
    random_rot_mat = Angles2Basis(rotation_angles)
    # print('random_rot_mat', random_rot_mat)

    new_rand_positions = Local2Global_Coord(random_rot_mat, translation_vecs, mesh_vertices)

    return new_rand_positions


def random_samples_selector(list_of_paths_incl_frames, numb_of_new_instances):
    '''
    This function will take the list of paths to the the directory where each individual frame's data is stored. It will then sample the list_of_paths to generate a new list of paths, which will include new paths to create random rotations on for data augmentation.
        input:
            list_of_paths_incl_frames --> list of all of the paths to where the frame data is actually held.
            numb_of_new_instances --> the number of how many new instances wanted to be added to the initial data set
        returns:
            list_of_paths_data_aug --> this will return a new list of paths where the data can be sampled from
    '''
    random_sample_of_paths = np.random.choice(list_of_paths_incl_frames, numb_of_new_instances, replace=True)

    list_of_paths_data_aug = list_of_paths_incl_frames + list(random_sample_of_paths)

    return list_of_paths_data_aug


def determine_voxel_max_shape_from_mesh_vertices(list_of_mesh_vertices, spacing=0.5):
    '''
    We are anticipating getting a list of the various mesh vertices, which have been set to local coordinate system, so the origin can be expected to be at [0,0,0].
    '''

    max_shape_vector = np.zeros(3)
    for mesh_verts in list_of_mesh_vertices:
        mesh_min_vec = np.min(mesh_verts, axis=0)
        mesh_min_vec = np.where(mesh_min_vec > 0, 0, mesh_min_vec)
        mesh_max_vec = np.max(mesh_verts, axis=0)
        mesh_max_vec = np.where(mesh_max_vec < 0, 0, mesh_max_vec)
        range_vec = mesh_max_vec - mesh_min_vec
        bins_vec = np.ceil(range_vec / spacing)

        if bins_vec[0] > max_shape_vector[0]:
            max_shape_vector[0] = bins_vec[0]

        if bins_vec[1] > max_shape_vector[1]:
            max_shape_vector[1] = bins_vec[1]

        if bins_vec[2] > max_shape_vector[2]:
            max_shape_vector[2] = bins_vec[2]


    return max_shape_vector


def matrix_padder_to_size(vox_mat, max_shape_vector):

    pad_mat = np.zeros((3, 2))
    # pad_mat = [[0, 0], [0, 0], [0, 0]]

    # print('------------\n\nMatrix shape: \n', vox_mat[item].shape, '\n\n-----------', '\n\n')
    # print('------------\n\nMax Shape Vector: \n', max_shape_vector, '\n\n-----------', '\n\n')
    if vox_mat.shape[0] < max_shape_vector[0]:
        pad_mat_0 = max_shape_vector[0] - vox_mat.shape[0]
        # print('\n\npad_mat_0:\n', pad_mat_0, '\n\n')
        if pad_mat_0 % 2 == 1:
            pad_mat[0, 0] = pad_mat_0 // 2 + 1
            pad_mat[0, 1] = pad_mat_0 // 2
        else:
            pad_mat[0, :] = pad_mat_0 / 2
            # pad_mat[0][1] = pad_mat_0 / 2

    if vox_mat.shape[1] < max_shape_vector[1]:
        pad_mat_1 = max_shape_vector[1] - vox_mat.shape[1]
        # print('\n\npad_mat_1:\n', pad_mat_1, '\n\n')
        if pad_mat_1 % 2 == 1:
            pad_mat[1, 0] = pad_mat_1 // 2 + 1
            pad_mat[1, 1] = pad_mat_1 // 2
        else:
            pad_mat[1, :] = pad_mat_1 / 2
            # pad_mat[1][1] = pad_mat_1 / 2

    if vox_mat.shape[2] < max_shape_vector[2]:
        pad_mat_2 = max_shape_vector[2] - vox_mat.shape[2]
        # print('\n\npad_mat_2:\n', pad_mat_2, '\n\n')
        if pad_mat_2 % 2 == 1:
            pad_mat[2, 0] = pad_mat_2 // 2 + 1
            pad_mat[2, 1] = pad_mat_2 // 2
        else:
            pad_mat[2, :] = pad_mat_2 / 2
            # pad_mat[2][1] = pad_mat_2 / 2

    # print('Pad mat: \n', pad_mat)

    return np.pad(vox_mat, pad_width=pad_mat.astype(int), mode='constant')

# ---------------------------------------------------------------
# This part of the file will deal with creating functions that will store the massive amounts of data


def generate_cali_storage_mat(list_of_path_to_frames, path_to_save_dir, save_file_name='cali_aug', compression=None):

    os.makedirs(path_to_save_dir, exist_ok=True)

    list_of_bones = ['Femur', 'Tibia']

    total_number_of_frames = len(list_of_path_to_frames) * len(list_of_bones)

    calibration_data = np.zeros((total_number_of_frames, 6))

    print('Calibration data shape: ', calibration_data.shape)

    calibration_file = h5py.File(os.path.join(path_to_save_dir, save_file_name + '.h5py'), 'w')
    cali_dset = calibration_file.create_dataset('cali_dset', data=calibration_data, compression=compression)

    ticker = 0
    for frame in list_of_path_to_frames:

        temp_cali_data = extract_calibration_data(os.path.abspath(os.sep.join(os.path.normpath(frame).split(os.sep)[:-1])))
        temp_cali_rot = Basis2Angles(temp_cali_data[0])
        temp_cali_trans = np.reshape(temp_cali_data[1], 3)

        interim_cali_array = np.hstack((temp_cali_rot, temp_cali_trans))

        cali_dset[2 * ticker: 2 * ticker + 2] = np.array([interim_cali_array, interim_cali_array])

        ticker += 1

    calibration_file.close()

    calibration_data = None

    return None


def generate_label_storage_mat(list_of_path_to_frames, path_to_save_dir, save_file_name='label_aug', compression=None):

    os.makedirs(path_to_save_dir, exist_ok=True)

    list_of_bones = ['Femur', 'Tibia']

    total_number_of_frames = len(list_of_path_to_frames) * len(list_of_bones)

    label_data = np.zeros((total_number_of_frames, 6))

    print('Label data shape: ', label_data.shape)

    label_file = h5py.File(os.path.join(path_to_save_dir, save_file_name + '.h5py'), 'w')
    label_dset = label_file.create_dataset('label_dset', data=label_data, compression=compression)

    ticker = 0
    for frame in list_of_path_to_frames:
        label_dset[2 * ticker: 2 * ticker + 2] = extract_labels_rot_trans_femur_tib_data(os.path.abspath(frame))


        ticker += 1

    label_file.close()

    label_data = None

    return None


def generate_image_storage_mat(list_of_path_to_frames, path_to_save_dir, save_file_name='images_aug', compression=None):

    os.makedirs(path_to_save_dir, exist_ok=True)

    list_of_bones = ['Femur', 'Tibia']

    total_number_of_frames = len(list_of_path_to_frames) * len(list_of_bones)

    image_data = np.zeros((total_number_of_frames, 2, 128, 128))

    print('Image data shape: ', image_data.shape)

    image_file = h5py.File(os.path.join(path_to_save_dir, save_file_name + '.h5py'), 'w')
    image_dset = image_file.create_dataset('image_dset', data=image_data, compression=compression)

    ticker = 0
    for frame in list_of_path_to_frames:
        image_dset[2 * ticker: 2 * ticker + 2] = extract_image_data(os.path.abspath(frame))

        if ticker % 500 == 0:
            print("Image: ", ticker)
        ticker += 1

    image_file.close()

    image_data = None

    return None


def generate_voxel_storage_mat(list_of_path_to_frames, path_to_save_dir, augmented_frames_number, save_file_name='voxels_aug', upload_set_size=35, compression='lzf', save_as_type='int8'):

    process_id = psutil.Process(os.getpid())

    date_time = datetime.datetime

    import time

    spacing = 0.5

    total_number_of_frames = len(list_of_path_to_frames)

    original_number_frames = total_number_of_frames - augmented_frames_number

    random_seed_array = np.random.randint(low=0, high=2**32, size=augmented_frames_number * 2)

    max_shape_vector = np.zeros(3)

    ticker = -1

    for path in list_of_path_to_frames:

        load_mesh_path_time = time.time()

        ticker += 1
        lat = os.path.abspath(os.sep.join(os.path.normpath(os.path.expanduser(path.replace('/Users/johndrago', '~', 1))).split(os.sep)[:-1]))
        lat_split = lat.split(os.sep)[-1]

        return_vox_tib_fib = [0, 0]

        if lat_split.lower() == 'lt':
            new_path = os.path.normpath(os.path.join(os.path.abspath(os.sep.join(os.path.normpath(lat).split(os.sep)[:-1])), 'stl'))

            # Left Femur
            LFemur = trimesh.load(os.path.join(new_path, 'LFemur.stl'))
            LFemur_PTS = np.array(pd.read_csv(os.path.join(new_path, 'LFemur_PTS.txt'), header=None))

            # Left Tibia
            LTibia = trimesh.load(os.path.join(new_path, 'LTibia.stl'))
            LTibia_PTS = np.array(pd.read_csv(os.path.join(new_path, 'LTibia_PTS.txt'), header=None))

            if ticker >= original_number_frames:

                seed_indexer = ticker - original_number_frames

                # print('Random')
                return_vox_tib_fib[0] = extract_stl_to_meshpoints(LFemur, LFemur_PTS, voxelize_dim=0.5, random_disp=True, random_seed=random_seed_array[2 * seed_indexer])
                return_vox_tib_fib[1] = extract_stl_to_meshpoints(LTibia, LTibia_PTS, voxelize_dim=0.5, random_disp=True, random_seed=random_seed_array[2 * seed_indexer + 1])
            else:
                # print('Not Random')
                return_vox_tib_fib[0] = extract_stl_to_meshpoints(LFemur, LFemur_PTS, voxelize_dim=0.5, random_disp=False)
                return_vox_tib_fib[1] = extract_stl_to_meshpoints(LTibia, LTibia_PTS, voxelize_dim=0.5, random_disp=False)

        if lat_split.lower() == 'rt':
            new_path = os.path.normpath(os.path.join(os.path.abspath(os.sep.join(os.path.normpath(lat).split(os.sep)[:-1])), 'stl'))

            # Right Femur
            RFemur = trimesh.load(os.path.join(new_path, 'RFemur.stl'))
            RFemur_PTS = np.array(pd.read_csv(os.path.join(new_path, 'RFemur_PTS.txt'), header=None))

            # Right Tibia
            RTibia = trimesh.load(os.path.join(new_path, 'RTibia.stl'))
            RTibia_PTS = np.array(pd.read_csv(os.path.join(new_path, 'RTibia_PTS.txt'), header=None))

            if ticker >= original_number_frames:

                seed_indexer = ticker - original_number_frames

                # print('Random')
                return_vox_tib_fib[0] = extract_stl_to_meshpoints(RFemur, RFemur_PTS, voxelize_dim=0.5, random_disp=True, random_seed=random_seed_array[2 * seed_indexer])
                return_vox_tib_fib[1] = extract_stl_to_meshpoints(RTibia, RTibia_PTS, voxelize_dim=0.5, random_disp=True, random_seed=random_seed_array[2 * seed_indexer + 1])
            else:
                # print('Not Random')
                return_vox_tib_fib[0] = extract_stl_to_meshpoints(RFemur, RFemur_PTS, voxelize_dim=0.5, random_disp=False)
                return_vox_tib_fib[1] = extract_stl_to_meshpoints(RTibia, RTibia_PTS, voxelize_dim=0.5, random_disp=False)

        for bone in return_vox_tib_fib:

            mesh_min_vec = np.min(bone, axis=0)
            mesh_min_vec = np.where(mesh_min_vec > 0, 0, mesh_min_vec)
            mesh_max_vec = np.max(bone, axis=0)
            mesh_max_vec = np.where(mesh_max_vec < 0, 0, mesh_max_vec)
            range_vec = mesh_max_vec - mesh_min_vec
            bins_vec = np.ceil(range_vec / spacing)

            if bins_vec[0] > max_shape_vector[0]:
                max_shape_vector[0] = int(bins_vec[0])

            if bins_vec[1] > max_shape_vector[1]:
                max_shape_vector[1] = int(bins_vec[1])

            if bins_vec[2] > max_shape_vector[2]:
                max_shape_vector[2] = int(bins_vec[2])

        time_to_load_mesh_path = time.time() - load_mesh_path_time

        if ticker % 200 == 0:
            print('----------------------------------------------------------------')
            print("Voxel mesh load: ", ticker)
            print('Time to load mesh path: ', round(time_to_load_mesh_path, 4), 'secs')
            if ticker >= original_number_frames:
                print('Random:', ticker, 'Seed: ', seed_indexer)
            else:
                print('Not Random: ', ticker)
            print('Memory Usage: ')
            print('\t', 'RSS: ', round(process_id.memory_info()[0] / 1e9, 3), 'GB')
            print('\t', 'VMS: ', round(process_id.memory_info()[1] / 1e9, 3), 'GB')

    vox_mat_shape = np.array([total_number_of_frames * 2, int(max_shape_vector[0]), int(max_shape_vector[1]), int(max_shape_vector[2])]).astype('int16')
    # vox_mat_shape = np.array([20000, int(max_shape_vector[0]), int(max_shape_vector[1]), int(max_shape_vector[2])]).astype('int16')
    print('\n' * 3)
    print('----------------------------------------------------------------')
    print('Voxel data shape: ', vox_mat_shape)
    print('----------------------------------------------------------------')
    print('\n' * 3)


    vox_file = h5py.File(os.path.join(path_to_save_dir, save_file_name + '.h5py'), 'w')

    vox_dset = vox_file.create_dataset('vox_dset', shape=vox_mat_shape, dtype=save_as_type, compression=compression)

    ticker1 = -1
    ticker3 = -1

    process_id.cpu_percent()

    upload_times = []

    for path_indx in range(int(np.ceil(len(list_of_path_to_frames) / upload_set_size))):
        ticker1 += 1

        num_sub_frames = len(list_of_path_to_frames[path_indx * upload_set_size: path_indx * upload_set_size + upload_set_size])

        vox_mat_sub = np.zeros((2 * num_sub_frames, vox_mat_shape[1], vox_mat_shape[2], vox_mat_shape[3])).astype(save_as_type)

        ticker2 = -1

        sequence_time = time.time()

        for path in list_of_path_to_frames[path_indx * upload_set_size: path_indx * upload_set_size + upload_set_size]:

            ticker2 += 1
            ticker3 += 1

            path_upload_time = time.time()

            lat = os.path.abspath(os.sep.join(os.path.normpath(os.path.expanduser(path.replace('/Users/johndrago', '~', 1))).split(os.sep)[:-1]))
            lat_split = lat.split(os.sep)[-1]

            return_vox_tib_fib = [0, 0]

            if lat_split.lower() == 'lt':
                new_path = os.path.normpath(os.path.join(os.path.abspath(os.sep.join(os.path.normpath(lat).split(os.sep)[:-1])), 'stl'))

                # Left Femur
                LFemur = trimesh.load(os.path.join(new_path, 'LFemur.stl'))
                LFemur_PTS = np.array(pd.read_csv(os.path.join(new_path, 'LFemur_PTS.txt'), header=None))

                # Left Tibia
                LTibia = trimesh.load(os.path.join(new_path, 'LTibia.stl'))
                LTibia_PTS = np.array(pd.read_csv(os.path.join(new_path, 'LTibia_PTS.txt'), header=None))


                if ticker3 >= original_number_frames:

                    seed_indexer = ticker3 - original_number_frames
                    # print('Random:', ticker3, 'Seed: ', seed_indexer)

                    return_vox_tib_fib[0] = extract_stl_to_meshpoints(LFemur, LFemur_PTS, voxelize_dim=0.5, random_disp=True, random_seed=random_seed_array[2 * seed_indexer])
                    return_vox_tib_fib[1] = extract_stl_to_meshpoints(LTibia, LTibia_PTS, voxelize_dim=0.5, random_disp=True, random_seed=random_seed_array[2 * seed_indexer + 1])
                else:

                    # print('Not Random: ', ticker3)
                    return_vox_tib_fib[0] = extract_stl_to_meshpoints(LFemur, LFemur_PTS, voxelize_dim=0.5, random_disp=False)
                    return_vox_tib_fib[1] = extract_stl_to_meshpoints(LTibia, LTibia_PTS, voxelize_dim=0.5, random_disp=False)

            if lat_split.lower() == 'rt':
                new_path = os.path.normpath(os.path.join(os.path.abspath(os.sep.join(os.path.normpath(lat).split(os.sep)[:-1])), 'stl'))

                # Right Femur
                RFemur = trimesh.load(os.path.join(new_path, 'RFemur.stl'))
                RFemur_PTS = np.array(pd.read_csv(os.path.join(new_path, 'RFemur_PTS.txt'), header=None))

                # Right Tibia
                RTibia = trimesh.load(os.path.join(new_path, 'RTibia.stl'))
                RTibia_PTS = np.array(pd.read_csv(os.path.join(new_path, 'RTibia_PTS.txt'), header=None))

                if ticker3 >= original_number_frames:

                    seed_indexer = ticker3 - original_number_frames
                    # print('Random:', ticker3, 'Seed: ', seed_indexer)

                    return_vox_tib_fib[0] = extract_stl_to_meshpoints(RFemur, RFemur_PTS, voxelize_dim=0.5, random_disp=True, random_seed=random_seed_array[2 * seed_indexer])
                    return_vox_tib_fib[1] = extract_stl_to_meshpoints(RTibia, RTibia_PTS, voxelize_dim=0.5, random_disp=True, random_seed=random_seed_array[2 * seed_indexer + 1])
                else:

                    # print('Not Random: ', ticker3)
                    return_vox_tib_fib[0] = extract_stl_to_meshpoints(RFemur, RFemur_PTS, voxelize_dim=0.5, random_disp=False)
                    return_vox_tib_fib[1] = extract_stl_to_meshpoints(RTibia, RTibia_PTS, voxelize_dim=0.5, random_disp=False)


            vox_dset_mat_1 = voxel_from_array(return_vox_tib_fib[0], spacing=spacing, mark_origin=True)


            vox_mat_sub[2 * ticker2] = matrix_padder_to_size(vox_dset_mat_1.astype(save_as_type), vox_mat_shape[1:]).astype(save_as_type)



            vox_dset_mat_2 = voxel_from_array(return_vox_tib_fib[1], spacing=spacing, mark_origin=True)



            vox_mat_sub[2 * ticker2 + 1] = matrix_padder_to_size(vox_dset_mat_2.astype(save_as_type), vox_mat_shape[1:]).astype(save_as_type)

            return_vox_tib_fib = None

            if ticker3 % upload_set_size == 0:
                print('----------------------------------------------------------------')
                print('Voxel: ', ticker3)

                if ticker3 >= original_number_frames:
                    print('Random:', ticker3, 'Seed: ', seed_indexer)
                else:
                    print('Not Random: ', ticker3)

                print('Voxel pad creation time: ', round(time.time() - path_upload_time, 4), 'secs')

        vox_mat_sub_time = time.time()
        vox_dset[2 * path_indx * upload_set_size: 2 * path_indx * upload_set_size + 2 * num_sub_frames] = vox_mat_sub.astype(save_as_type)
        vox_mat_sub_time_finish = time.time()
        print('--')
        print('Time to upload vox_mat_sub per instance: ', round((vox_mat_sub_time_finish - vox_mat_sub_time) / (num_sub_frames * 2), 4), 'secs')
        print('Time to upload upload_size_set total: ', round((vox_mat_sub_time_finish - vox_mat_sub_time), 4), 'secs')
        print('Total time to create and upload upload_size_set: ', round((vox_mat_sub_time_finish - sequence_time), 4), 'secs')
        upload_times.append(vox_mat_sub_time_finish - sequence_time)
        print('Average +/- SD upload times: ', round(np.mean(upload_times), 3), '+/-', round(np.std(upload_times), 3), 'secs')

        if (vox_mat_sub_time_finish - sequence_time) > 400:
            vox_file.close()
            vox_mat_sub = None
            print('\n' * 1)
            print('Pausing 30 seconds')
            time.sleep(30)
            print('Re-opening vox_file')
            vox_file = h5py.File(os.path.join(path_to_save_dir, save_file_name + '.h5py'), 'r+')
            vox_dset = vox_file['vox_dset']
            print('\n' * 1)

        vox_mat_sub = None

        print('--')
        print('Memory Usage: ')
        print('\t', 'RSS: ', round(process_id.memory_info()[0] / 1e9, 3), 'GB')
        print('\t', 'VMS: ', round(process_id.memory_info()[1] / 1e9, 3), 'GB')
        print('CPU percent for process: ', process_id.cpu_percent(), '%')
        print('CPU percent per cpu: ', psutil.cpu_percent(percpu=True))
        print('--')
        print('Date/Time: ', date_time.now())

    vox_file.close()

    return None


























# ---------------------------------------------------------------
# ---------------------------------------------------------------



if __name__ == '__main__':
    # dict_of_acts = generate_dict_of_acts_with_patients()
    # dict_of_paths = generate_dict_path_to_frames(dict_of_acts)
    # list_of_frames = generate_comprehensive_list_of_frames(dict_of_paths)
    # list_of_frames = sorted(list_of_frames)

    # augmented_frames_number = 10000

    # data_aug_list_of_frames = random_samples_selector(list_of_frames, augmented_frames_number)

    # aug_dict = {}
    # aug_dict['all_paths'] = data_aug_list_of_frames
    # aug_dict['numb_aug_frames'] = augmented_frames_number
    # aug_dict['list_of_original_frames'] = list_of_frames

    # aug_file = open(os.path.join(save_dir, 'aug_data' + '.pkl'), 'wb')

    # pickle.dump(aug_dict, aug_file)

    # aug_file.close()

    save_dir = os.path.abspath(os.path.join(os.path.expanduser('~/fluoro/data/compilation')))

    aug_file = open(os.path.join(save_dir, 'aug_data' + '.pkl'), 'rb')
    aug_dict = pickle.load(aug_file)

    data_aug_list_of_frames = aug_dict['all_paths']
    augmented_frames_number = aug_dict['numb_aug_frames']

    aug_file.close()


    # generate_cali_storage_mat(data_aug_list_of_frames, save_dir)
    # generate_label_storage_mat(data_aug_list_of_frames, save_dir)
    # generate_image_storage_mat(data_aug_list_of_frames, save_dir)
    generate_voxel_storage_mat(data_aug_list_of_frames, save_dir, augmented_frames_number)

    # generate_voxel_storage_mat_test(data_aug_list_of_frames[:1000], save_dir, augmented_frames_number=900, save_file_name='voxels_test2')














# ---------------------------------------------------------------


# def generate_voxel_storage_mat(list_of_path_to_frames, path_to_save_dir, augmented_frames_number, save_file_name='voxels_aug', compression='lzf', save_as_type='int8'):

#     import time

#     spacing = 0.5

#     total_number_of_frames = len(list_of_path_to_frames)

#     original_number_frames = total_number_of_frames - augmented_frames_number

#     random_seed_array = np.random.randint(low=0, high=2**32, size=augmented_frames_number * 2)

#     max_shape_vector = np.zeros(3)

#     ticker = -1

#     for path in list_of_path_to_frames:

#         load_mesh_path_time = time.time()

#         ticker += 1
#         lat = os.path.abspath(os.sep.join(os.path.normpath(path).split(os.sep)[:-1]))
#         lat_split = lat.split(os.sep)[-1]

#         return_vox_tib_fib = [0, 0]

#         if lat_split.lower() == 'lt':
#             new_path = os.path.normpath(os.path.join(os.path.abspath(os.sep.join(os.path.normpath(lat).split(os.sep)[:-1])), 'stl'))

#             # Left Femur
#             LFemur = trimesh.load(os.path.join(new_path, 'LFemur.stl'))
#             LFemur_PTS = np.array(pd.read_csv(os.path.join(new_path, 'LFemur_PTS.txt'), header=None))

#             # Left Tibia
#             LTibia = trimesh.load(os.path.join(new_path, 'LTibia.stl'))
#             LTibia_PTS = np.array(pd.read_csv(os.path.join(new_path, 'LTibia_PTS.txt'), header=None))

#             if ticker >= original_number_frames:

#                 seed_indexer = ticker - original_number_frames

#                 # print('Random')
#                 return_vox_tib_fib[0] = extract_stl_to_meshpoints(LFemur, LFemur_PTS, voxelize_dim=0.5, random_disp=True, random_seed=random_seed_array[2 * seed_indexer])
#                 return_vox_tib_fib[1] = extract_stl_to_meshpoints(LTibia, LTibia_PTS, voxelize_dim=0.5, random_disp=True, random_seed=random_seed_array[2 * seed_indexer + 1])
#             else:
#                 # print('Not Random')
#                 return_vox_tib_fib[0] = extract_stl_to_meshpoints(LFemur, LFemur_PTS, voxelize_dim=0.5, random_disp=False)
#                 return_vox_tib_fib[1] = extract_stl_to_meshpoints(LTibia, LTibia_PTS, voxelize_dim=0.5, random_disp=False)

#         if lat_split.lower() == 'rt':
#             new_path = os.path.normpath(os.path.join(os.path.abspath(os.sep.join(os.path.normpath(lat).split(os.sep)[:-1])), 'stl'))

#             # Right Femur
#             RFemur = trimesh.load(os.path.join(new_path, 'RFemur.stl'))
#             RFemur_PTS = np.array(pd.read_csv(os.path.join(new_path, 'RFemur_PTS.txt'), header=None))

#             # Right Tibia
#             RTibia = trimesh.load(os.path.join(new_path, 'RTibia.stl'))
#             RTibia_PTS = np.array(pd.read_csv(os.path.join(new_path, 'RTibia_PTS.txt'), header=None))

#             if ticker >= original_number_frames:

#                 seed_indexer = ticker - original_number_frames

#                 # print('Random')
#                 return_vox_tib_fib[0] = extract_stl_to_meshpoints(RFemur, RFemur_PTS, voxelize_dim=0.5, random_disp=True, random_seed=random_seed_array[2 * seed_indexer])
#                 return_vox_tib_fib[1] = extract_stl_to_meshpoints(RTibia, RTibia_PTS, voxelize_dim=0.5, random_disp=True, random_seed=random_seed_array[2 * seed_indexer + 1])
#             else:
#                 # print('Not Random')
#                 return_vox_tib_fib[0] = extract_stl_to_meshpoints(RFemur, RFemur_PTS, voxelize_dim=0.5, random_disp=False)
#                 return_vox_tib_fib[1] = extract_stl_to_meshpoints(RTibia, RTibia_PTS, voxelize_dim=0.5, random_disp=False)

#         for bone in return_vox_tib_fib:

#             mesh_min_vec = np.min(bone, axis=0)
#             mesh_min_vec = np.where(mesh_min_vec > 0, 0, mesh_min_vec)
#             mesh_max_vec = np.max(bone, axis=0)
#             mesh_max_vec = np.where(mesh_max_vec < 0, 0, mesh_max_vec)
#             range_vec = mesh_max_vec - mesh_min_vec
#             bins_vec = np.ceil(range_vec / spacing)

#             if bins_vec[0] > max_shape_vector[0]:
#                 max_shape_vector[0] = int(bins_vec[0])

#             if bins_vec[1] > max_shape_vector[1]:
#                 max_shape_vector[1] = int(bins_vec[1])

#             if bins_vec[2] > max_shape_vector[2]:
#                 max_shape_vector[2] = int(bins_vec[2])

#         time_to_load_mesh_path = time.time() - load_mesh_path_time

#         if ticker % 200 == 0:
#             print('\n')
#             print("Voxel mesh load: ", ticker)
#             print('Time to load mesh path: ', time_to_load_mesh_path)
#             if ticker >= original_number_frames:
#                 print('Random')
#             else:
#                 print('Not Random')

#     vox_mat_shape = np.array([total_number_of_frames * 2, int(max_shape_vector[0]), int(max_shape_vector[1]), int(max_shape_vector[2])]).astype('int16')

#     print('Voxel data shape: ', vox_mat_shape)

#     vox_file = h5py.File(os.path.join(path_to_save_dir, save_file_name + '.h5py'), 'w')

#     vox_dset = vox_file.create_dataset('vox_dset', shape=vox_mat_shape, dtype=save_as_type, compression=None)

#     ticker2 = -1

#     for path in list_of_path_to_frames:

#         time_to_vox_pad = time.time()

#         ticker2 += 1

#         lat = os.path.abspath(os.sep.join(os.path.normpath(path).split(os.sep)[:-1]))
#         lat_split = lat.split(os.sep)[-1]

#         return_vox_tib_fib = [0, 0]

#         if lat_split.lower() == 'lt':
#             new_path = os.path.normpath(os.path.join(os.path.abspath(os.sep.join(os.path.normpath(lat).split(os.sep)[:-1])), 'stl'))

#             # Left Femur
#             LFemur = trimesh.load(os.path.join(new_path, 'LFemur.stl'))
#             LFemur_PTS = np.array(pd.read_csv(os.path.join(new_path, 'LFemur_PTS.txt'), header=None))

#             # Left Tibia
#             LTibia = trimesh.load(os.path.join(new_path, 'LTibia.stl'))
#             LTibia_PTS = np.array(pd.read_csv(os.path.join(new_path, 'LTibia_PTS.txt'), header=None))

#             if ticker2 >= original_number_frames:

#                 seed_indexer = ticker2 - original_number_frames

#                 return_vox_tib_fib[0] = extract_stl_to_meshpoints(LFemur, LFemur_PTS, voxelize_dim=0.5, random_disp=True, random_seed=random_seed_array[2 * seed_indexer])
#                 return_vox_tib_fib[1] = extract_stl_to_meshpoints(LTibia, LTibia_PTS, voxelize_dim=0.5, random_disp=True, random_seed=random_seed_array[2 * seed_indexer + 1])
#             else:
#                 return_vox_tib_fib[0] = extract_stl_to_meshpoints(LFemur, LFemur_PTS, voxelize_dim=0.5, random_disp=False)
#                 return_vox_tib_fib[1] = extract_stl_to_meshpoints(LTibia, LTibia_PTS, voxelize_dim=0.5, random_disp=False)

#         if lat_split.lower() == 'rt':
#             new_path = os.path.normpath(os.path.join(os.path.abspath(os.sep.join(os.path.normpath(lat).split(os.sep)[:-1])), 'stl'))

#             # Right Femur
#             RFemur = trimesh.load(os.path.join(new_path, 'RFemur.stl'))
#             RFemur_PTS = np.array(pd.read_csv(os.path.join(new_path, 'RFemur_PTS.txt'), header=None))

#             # Right Tibia
#             RTibia = trimesh.load(os.path.join(new_path, 'RTibia.stl'))
#             RTibia_PTS = np.array(pd.read_csv(os.path.join(new_path, 'RTibia_PTS.txt'), header=None))

#             if ticker2 >= original_number_frames:

#                 seed_indexer = ticker2 - original_number_frames

#                 return_vox_tib_fib[0] = extract_stl_to_meshpoints(RFemur, RFemur_PTS, voxelize_dim=0.5, random_disp=True, random_seed=random_seed_array[2 * seed_indexer])
#                 return_vox_tib_fib[1] = extract_stl_to_meshpoints(RTibia, RTibia_PTS, voxelize_dim=0.5, random_disp=True, random_seed=random_seed_array[2 * seed_indexer + 1])
#             else:
#                 return_vox_tib_fib[0] = extract_stl_to_meshpoints(RFemur, RFemur_PTS, voxelize_dim=0.5, random_disp=False)
#                 return_vox_tib_fib[1] = extract_stl_to_meshpoints(RTibia, RTibia_PTS, voxelize_dim=0.5, random_disp=False)

#         time_vox_mat = time.time()

#         vox_dset_mat_1 = voxel_from_array(return_vox_tib_fib[0], spacing=spacing, mark_origin=True)

#         print('Make voxel array 1: ', time.time() - time_vox_mat)


#         vox_dset[2 * ticker2] = matrix_padder_to_size(vox_dset_mat_1.astype(save_as_type), vox_mat_shape[1:]).astype(save_as_type)

#         print('Upload voxel array 1: ', time.time() - time_vox_mat)


#         time_vox_mat = time.time()


#         vox_dset_mat_2 = voxel_from_array(return_vox_tib_fib[1], spacing=spacing, mark_origin=True)

#         print('Make voxel array 2: ', time.time() - time_vox_mat)


#         vox_dset[2 * ticker2 + 1] = matrix_padder_to_size(vox_dset_mat_2.astype(save_as_type), vox_mat_shape[1:]).astype(save_as_type)

#         print('Upload voxel array 2: ', time.time() - time_vox_mat)
#         print('\n')

#         return_vox_tib_fib = [0, 0]

#         time_to_vox_pad_load = time.time() - time_to_vox_pad

#         if ticker2 % 20 == 0:
#             print("Voxel: ", ticker2)
#             print('Time to vox pad load: ', time_to_vox_pad_load)


#     vox_file.close()

#     return None
