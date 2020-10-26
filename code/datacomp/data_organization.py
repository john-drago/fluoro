'''
This file will organize the data in the 'fluoro/data' folder. It will organize the stl files and the photos into large matrices to allow for training.
'''

import os
from coord_change import Global2Local_Coord, Basis2Angles
import scipy.io as sio
import skimage
import numpy as np
import trimesh
import pandas as pd
import sys
from scipy import ndimage
import h5py
import pickle


from h5py_multidimensional_array import variable_matrix_storer, variable_matrix_padder
# from voxel_graph import simple_voxel_graph

print('\n', '\n', '\n')


# this should be located within the */fluoro/data folder
top_level_dir = '/Users/johndrago/fluoro/data'


def create_dir_path(*name):
    '''
    This function will take in a series of directory paths, i.e. create_dir_path('foo','bar','egg','fyegg') and will output a file path from left to right:
    '/foo/bar/egg/fyegg'
    '''
    file_path = ''
    ticker = 0
    for direct in name:
        ticker += 1
        if not ticker == len(name):
            file_path = file_path + direct + '/'
        else:
            file_path = file_path + direct
    return os.path.normpath(file_path)


def generate_dict_of_acts_with_patients():
    '''
    This function will generate a dictionary of the different activities and all of the patients who did that activity.

    The final format will be a dictionary where keys are the activities and the values are the different patients who did the activity.

    Assuming that we are in path: */fluoro/data
    '''

    activity_list = []
    pt_dict = {}

    for direct1 in os.listdir(create_dir_path(top_level_dir)):
        if direct1 != '.DS_Store' and direct1 != 'compilation' and direct1 != 'prediction':
            activity_list.append(direct1)

            # print(direct1)

            list_of_pts_for_act = []
            for direct2 in os.listdir(create_dir_path(top_level_dir, direct1)):


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
            dir_to_pt = create_dir_path(top_level_dir, act, pt)
            for direct3 in os.listdir(dir_to_pt):
                list_of_frames = []
                if direct3 != '.DS_Store' and direct3 != 'stl':
                    dir_to_frames = create_dir_path(top_level_dir, act, pt, direct3)
                    # print(dir_to_frames)
                    for frame in os.listdir(dir_to_frames):
                        if (frame != '.DS_Store') and frame != 'cali':
                            list_of_frames.append(frame)
                    path_to_frames_dict[dir_to_frames] = list_of_frames
    return path_to_frames_dict


def extract_calibration_data(path_to_cali):
    '''
    This function will return the R12 and V12 variables from the reg2fl***.mat file in the "data/activity/patient/laterality/cali" folder.

    extract_calibration_data(path_to_cali)

        input: expects a path to the directory where the cali frame is located

        output: will output an array of the form: [ R12, V12 ]

    '''
    cali_str = 'cali'
    for fle in os.listdir(create_dir_path(path_to_cali, cali_str)):
        if fle[0:6].lower() == 'reg2fl':
            fluoro_file = sio.loadmat(create_dir_path(path_to_cali, cali_str, fle))
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
    image_array = np.zeros((2, resize_shape[0], resize_shape[1]))

    for fle in os.listdir(os.path.normpath(path_to_frame)):
        if fle[-4:] == '.png' and fle[0:2] == 'F1':
            image_load = skimage.io.imread(create_dir_path(path_to_frame, fle))
            image_resize = skimage.transform.resize(image_load, resize_shape, anti_aliasing=True)
            image_gray = skimage.color.rgb2gray(image_resize)
            # image_array[0] = image_gray.reshape(resize_shape[0] * resize_shape[1])
            image_array[0] = image_gray
            # print(type(image_gray))
        elif fle[-4:] == '.png' and fle[0:2] == 'F2':
            image_load = skimage.io.imread(create_dir_path(path_to_frame, fle))
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
            results_file = sio.loadmat(create_dir_path(path_to_frame, fle))

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




def extract_stl_to_voxel_trimesh(mesh_obj, PTS_file, voxelize_dim=0.5):
    '''
    In sum, this function will:

    1) take an STL file loaded as a mesh object and take the PTS file loaded as a pandas object

    2) using the PTS file, determine local coordinate frame and shift STL point cloud to new local coordinate frame

    3) voxelize the vertices of the point cloud to binary, depending on if a vertex would be in the corresponding voxel. Uses trimesh functionality to perform voxelization

    4) return an array of both 3D voxel models for loaded model


    function extract_stl_to_voxel_trimesh(path_to_frame, voxelize_dim=0.5)
        input:
            mesh_obj --> loaded trimesh mesh object (stl file)
            PTS_file --> loaded pandas
            voxelize_dim --> the scale of creating new voxel map

        output:
            3D binary voxel model as an array


    This function assumes it will be passed the loaded trimesh mesh as an argument. This function will produce an a NumPy array of the binary voxel data.

    In doing so, this function will also translate the points in the stl file to the local coordinate frame defined through the PTS files:

    PTS file: ---> defines the new X and Z directions of the local coordinate system
        X Coordinate system: From PTS row 1 to PTS row 0
        Z Coordinate system: From PTS row 3 to PTS row 2

    The origin of the new coordinate system is defined to be halfway between the two anatomical points, which demarcate the x-axis.

    From these two coordinates, we can determine the Y axis, via the cross product of the unit vectors: Y = cross(z,x)
    '''

    PTS_file = np.array(PTS_file)


    X_vec = np.array(PTS_file[0, :] - PTS_file[1, :])
    Z_vec_pre = np.array(PTS_file[2, :] - PTS_file[3, :])

    Y_vec = np.cross(Z_vec_pre, X_vec)

    # We have to do the second cross, because we cannot a priori guarantee that the X and Z unit vectors are orthogonal. Once we generate an orthogonal Y unit vector, we can regenerate the Z unit vector based on the X and Y unit vectors to create a true orthonormal basis

    Z_vec = np.cross(X_vec, Y_vec)

    x_unit = X_vec / np.linalg.norm(X_vec)
    y_unit = Y_vec / np.linalg.norm(Y_vec)
    z_unit = Z_vec / np.linalg.norm(Z_vec)



    rot_mat = np.array([
        [x_unit[0], y_unit[0], z_unit[0]],
        [x_unit[1], y_unit[1], z_unit[1]],
        [x_unit[2], y_unit[2], z_unit[2]]])

    origin_mesh_local = np.array(PTS_file[0:2, :].mean(axis=0))

    STL_local_coord = Global2Local_Coord(rot_mat, origin_mesh_local, mesh_obj.vertices)
    STL_local_coord_mesh = trimesh.Trimesh(vertices=STL_local_coord, faces=mesh_obj.faces)

    STL_voxelized = STL_local_coord_mesh.voxelized(voxelize_dim)

    vox_matrix = STL_voxelized.matrix

    return_vox_matrix = vox_matrix.astype(int)


    return return_vox_matrix


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



def extract_stl_to_voxel(mesh_obj, PTS_file, voxelize_dim=0.5):
    '''
    In sum, this function will:

    1) take an STL file loaded as a mesh object and take the PTS file loaded as a pandas object

    2) using the PTS file, determine local coordinate frame and shift STL point cloud to new local coordinate frame

    3) voxelize the vertices of the point cloud to binary, depending on if a vertex would be in the corresponding voxel. Uses in-house algorithm.

    4) return an array of both 3D voxel models for loaded model


    function extract_stl_to_voxel(path_to_frame, voxelize_dim=0.5)
        input:
            mesh_obj --> loaded trimesh mesh object (stl file)
            PTS_file --> loaded pandas
            voxelize_dim --> the scale of creating new voxel map

        output:
            3D binary voxel model as an array


    This function assumes it will be passed the loaded trimesh mesh as an argument. This function will produce an a NumPy array of the binary voxel data.

    In doing so, this function will also translate the points in the stl file to the local coordinate frame defined through the PTS files:

    PTS file: ---> defines the new X and Z directions of the local coordinate system
        X Coordinate system: From PTS row 1 to PTS row 0
        Z Coordinate system: From PTS row 3 to PTS row 2

    The origin of the new coordinate system is defined to be halfway between the two anatomical points, which demarcate the x-axis.

    From these two coordinates, we can determine the Y axis, via the cross product of the unit vectors: Y = cross(z,x)
    '''

    PTS_file = np.array(PTS_file)


    X_vec = np.array(PTS_file[0, :] - PTS_file[1, :])
    Z_vec_pre = np.array(PTS_file[2, :] - PTS_file[3, :])

    Y_vec = np.cross(Z_vec_pre, X_vec)

    # We have to do the second cross, because we cannot a priori guarantee that the X and Z unit vectors are orthogonal. Once we generate an orthogonal Y unit vector, we can regenerate the Z unit vector based on the X and Y unit vectors to create a true orthonormal basis

    Z_vec = np.cross(X_vec, Y_vec)

    x_unit = X_vec / np.linalg.norm(X_vec)
    y_unit = Y_vec / np.linalg.norm(Y_vec)
    z_unit = Z_vec / np.linalg.norm(Z_vec)


    rot_mat = np.array([
        [x_unit[0], y_unit[0], z_unit[0]],
        [x_unit[1], y_unit[1], z_unit[1]],
        [x_unit[2], y_unit[2], z_unit[2]]])

    origin_mesh_local = np.array(PTS_file[0:2, :].mean(axis=0))

    verts_local_coord = Global2Local_Coord(rot_mat, origin_mesh_local, mesh_obj.vertices)

    bin_mat = voxel_from_array(verts_local_coord, spacing=voxelize_dim, mark_origin=True)

    return bin_mat.astype('int8')



def stl_located(path_to_frames, voxelized_dim=0.5):
    '''
    This function will take the path to the directory, where the frames of a given motion are stored:

    */fluoro/data/activity/patient/laterality

    It will then use the extract_stl_to_voxel function to generate an array of the voxelized data for the tibia and fibia.

    input:
        path_to_frames --> where the frames are located for each motion
        voxelized_dim --> scale of new voxel map


    output
        array of the two 3D voxel models

            [ femur 3D voxel model, tibia 3D voxel model ]
    '''
    lat = os.path.normpath(path_to_frames)
    lat_split = lat.split(os.sep)[-1]
    return_vox_tib_fib = [0, 0]

    if lat_split.lower() == 'lt':
        new_path = create_dir_path(os.path.normpath(lat[:-3]), 'stl')

        # Left Femur

        LFemur = trimesh.load(create_dir_path(new_path, 'LFemur.stl'))
        LFemur_PTS = pd.read_csv(create_dir_path(new_path, 'LFemur_PTS.txt'), header=None)
        LFemur_PTS_np = np.array(LFemur_PTS)

        return_vox_tib_fib[0] = extract_stl_to_voxel(LFemur, LFemur_PTS_np)

        # Left Tibia

        LTibia = trimesh.load(create_dir_path(new_path, 'LTibia.stl'))

        LTibia_PTS = pd.read_csv(create_dir_path(new_path, 'LTibia_PTS.txt'), header=None)
        LTibia_PTS_np = np.array(LTibia_PTS)


        return_vox_tib_fib[1] = extract_stl_to_voxel(LTibia, LTibia_PTS_np)

    elif lat_split.lower() == 'rt':
        new_path = create_dir_path(os.path.normpath(lat[:-3]), 'stl')

        # Right Femur

        RFemur = trimesh.load(create_dir_path(new_path, 'RFemur.stl'))

        RFemur_PTS = pd.read_csv(create_dir_path(new_path, 'RFemur_PTS.txt'), header=None)
        RFemur_PTS_np = np.array(RFemur_PTS)

        return_vox_tib_fib[0] = extract_stl_to_voxel(RFemur, RFemur_PTS_np)


        # Right Tibia

        RTibia = trimesh.load(create_dir_path(new_path, 'RTibia.stl'))

        RTibia_PTS = pd.read_csv(create_dir_path(new_path, 'RTibia_PTS.txt'), header=None)
        RTibia_PTS_np = np.array(RTibia_PTS)

        return_vox_tib_fib[1] = extract_stl_to_voxel(RTibia, RTibia_PTS_np)


    return return_vox_tib_fib




def voxel_binary_to_distance_transform(voxel_binary):
    '''
    This function will take in a binary voxel data set, and it will convert it to a distance transform (https://en.wikipedia.org/wiki/Distance_transform)

    '''
    return ndimage.distance_transform_edt(voxel_binary)


def directory_path_to_frames(path_to_frame):
    '''
    This function takes a path to where the frames are located, and it returns a list of the paths to each frame directory:

    input:
        path_to_frame --> */fluoro/data/act/patient/laterality

    output:
        path_to_frame_dir --> */fluoro/data/act/patient/laterality/frame
    '''
    list_of_frames = []
    temp_list_of_frames = os.listdir(path_to_frame)

    for frme in temp_list_of_frames:
        if frme != '.DS_Store' and frme != 'cali':
            list_of_frames.append(create_dir_path(path_to_frame, frme))

    list_of_frames.sort()

    return list_of_frames



def generate_and_save_cali_compilation_matrix(list_of_path_to_frames, dict_of_path_to_frames, path_to_save_compilation):
    '''
    This function will take a sorted list of the paths to where the frames are held, i.e.:

    */fluoro/data/activity/patient/laterality

    It will also take a dictionary, where the list of paths serve as the keys, and the value is a list corresponding to the frames for a given path to frames.

    e.g. */fluoro/data/activity/patient/laterality/frame

    It will then generate a matrix of the following dimensions and characteristics:


    Rotation matrix and translation vector describing the relative positioning of the two fluoroscopes.

    shape = 9 + 3, frames*2.
    '''

    os.makedirs(path_to_save_compilation, exist_ok=True)

    list_of_path_to_frames.sort()



    list_of_bones = ['Femur', 'Tibia']

    total_number_of_frames = 0

    for frame_dir1 in sorted(list_of_path_to_frames):
        for frme1 in sorted(dict_of_path_to_frames[frame_dir1]):
            total_number_of_frames += 1

    number_of_frames_to_match = total_number_of_frames * len(list_of_bones)

    print('Number of frames to match:\t', number_of_frames_to_match)

    calibration_data_len9_mat = np.zeros((number_of_frames_to_match, 12))
    calibration_data_len3_mat = np.zeros((number_of_frames_to_match, 6))

    ticker = -1

    calibration_file = h5py.File(create_dir_path(path_to_save_compilation, 'calibration.h5py'), 'w')


    for frame_dir2 in sorted(list_of_path_to_frames):
        for frme2 in sorted(dict_of_path_to_frames[frame_dir2]):
            print('\n\n', create_dir_path(frame_dir2, frme2))

            ticker += 1

            temp_cali_data = extract_calibration_data(frame_dir2)

            temp_cali_len9_rot = np.reshape(temp_cali_data[0], 9)
            temp_cali_len3_rot = np.reshape(Basis2Angles(temp_cali_data[0]), 3)

            temp_cali_trans = np.reshape(temp_cali_data[1], 3)

            interim_array_holder_len9 = np.hstack((temp_cali_len9_rot, temp_cali_trans))
            interim_array_holder_len3 = np.hstack((temp_cali_len3_rot, temp_cali_trans))

            calibration_data_len9_mat[2 * ticker:2 * ticker + 2] = np.array([interim_array_holder_len9, interim_array_holder_len9])
            calibration_data_len3_mat[2 * ticker:2 * ticker + 2] = np.array([interim_array_holder_len3, interim_array_holder_len3])

    cali_9m = calibration_file.create_dataset('cali_len9_rot', data=calibration_data_len9_mat)
    cali_3m = calibration_file.create_dataset('cali_len3_rot', data=calibration_data_len3_mat)

    calibration_file.close()

    return cali_9m, cali_3m


def generate_and_save_image_compilation_matrix(list_of_path_to_frames, dict_of_path_to_frames, path_to_save_compilation):
    '''
    This function will take a sorted list of the paths to where the frames are held, i.e.:

    */fluoro/data/activity/patient/laterality

    It will also take a dictionary, where the list of paths serve as the keys, and the value is a list corresponding to the frames for a given path to frames.

    e.g. */fluoro/data/activity/patient/laterality/frame

    It will then generate a matrix of the following dimensions and characteristics:

    - Matrix of two png images, converted to gray scale and downsized.

    shape = frames * 2, 2, 128, 128

    '''
    os.makedirs(path_to_save_compilation, exist_ok=True)

    list_of_path_to_frames.sort()

    list_of_bones = ['Femur', 'Tibia']

    total_number_of_frames = 0

    for frame_dir1 in sorted(list_of_path_to_frames):
        for frme1 in sorted(dict_of_path_to_frames[frame_dir1]):
            total_number_of_frames += 1

    number_of_frames_to_match = total_number_of_frames * len(list_of_bones)

    print('Number of frames to match:\t', number_of_frames_to_match)

    image_data_mat = np.random.rand(number_of_frames_to_match, 2, 128, 128)

    ticker = -1

    image_file = h5py.File(create_dir_path(path_to_save_compilation, 'images.h5py'), 'w')

    for frame_dir2 in sorted(list_of_path_to_frames):
        for frme2 in sorted(dict_of_path_to_frames[frame_dir2]):
            print('\n\n', create_dir_path(frame_dir2, frme2))

            ticker += 1

            temp_image_data = extract_image_data(create_dir_path(frame_dir2, frme2))

            image_data_mat[2 * ticker:2 * ticker + 2, :, :, :] = temp_image_data

    image_dset = image_file.create_dataset('image_dset', data=image_data_mat)

    image_file.close()

    return image_dset


def generate_and_save_label_compilation_matrix(list_of_path_to_frames, dict_of_path_to_frames, path_to_save_compilation):
    '''
    This function will take a sorted list of the paths to where the frames are held, i.e.:

    */fluoro/data/activity/patient/laterality

    It will also take a dictionary, where the list of paths serve as the keys, and the value is a list corresponding to the frames for a given path to frames.

    e.g. */fluoro/data/activity/patient/laterality/frame

    It will then generate a matrix of the following dimensions and characteristics:

    Labels for each frame; comprised of rotation vector (theta, phi, psi) and translation vector (x,y,z) describing where the local coordinate system of the bone of interest has shifted to.

    shape:
        total_number_of_frames * 2, 6

    '''
    os.makedirs(path_to_save_compilation, exist_ok=True)

    list_of_path_to_frames.sort()

    list_of_bones = ['Femur', 'Tibia']

    total_number_of_frames = 0

    for frame_dir1 in sorted(list_of_path_to_frames):
        for frme1 in sorted(dict_of_path_to_frames[frame_dir1]):
            total_number_of_frames += 1

    print('Total number of frames:\t', total_number_of_frames)

    number_of_frames_to_match = total_number_of_frames * len(list_of_bones)

    print('Number of frames to match:\t', number_of_frames_to_match)

    label_data_mat = np.random.rand(number_of_frames_to_match, 6)

    ticker = -1

    label_file = h5py.File(create_dir_path(path_to_save_compilation, 'labels.h5py'), 'w')

    for frame_dir2 in sorted(list_of_path_to_frames):
        for frme2 in sorted(dict_of_path_to_frames[frame_dir2]):
            print(create_dir_path(frame_dir2, frme2))

            ticker += 1

            temp_label_data = extract_labels_rot_trans_femur_tib_data(create_dir_path(frame_dir2, frme2))

            label_data_mat[2 * ticker: 2 * ticker + 2, :] = temp_label_data

    labels_dset = label_file.create_dataset('labels_dset', data=label_data_mat)

    label_file.close()

    return labels_dset


def generate_and_save_mesh_voxel_compilation_matrix(list_of_path_to_frames, dict_of_path_to_frames, path_to_save_compilation, storage_file_name=None, save_as_type='uint8'):
    '''
    This function will take a sorted list of the paths to where the frames are held, i.e.:

    */fluoro/data/activity/patient/laterality

    It will also take a dictionary, where the list of paths serve as the keys, and the value is a list corresponding to the frames for a given path to frames.

    e.g. */fluoro/data/activity/patient/laterality/frame

    It will then generate a matrix of the following dimensions and characteristics:

    Matrix of .stl files representing a binary voxel dataset, where the 1 voxels correspond to a mesh vertex being located there. shape = x, y, z, frames*2 . Frames will correspond to the amount of paths to frames there are present, and there will be an additional factor of 2 for both the femur and the tibia.

    shape:
        total_number_of_frames * 2, x,y,z of the voxelized mesh (in binary, with 1s for location of vertex and 0 when no vertex present)

    '''
    if not storage_file_name:
        storage_file_name = 'voxels'

    os.makedirs(path_to_save_compilation, exist_ok=True)

    list_of_path_to_frames.sort()

    list_of_bones = ['Femur', 'Tibia']

    total_number_of_frames = 0

    for frame_dir1 in sorted(list_of_path_to_frames):
        for frme1 in sorted(dict_of_path_to_frames[frame_dir1]):
            total_number_of_frames += 1

    print('Total number of frames:\t', total_number_of_frames)

    number_of_frames_to_match = total_number_of_frames * len(list_of_bones)

    print('Number of frames to match:\t', number_of_frames_to_match)

    vox_data_mat = [0] * number_of_frames_to_match

    ticker = -1

    # vox_file = h5py.File(create_dir_path(path_to_save_compilation, 'voxels.h5py'), 'w')

    for frame_dir2 in sorted(list_of_path_to_frames):

        temp_vox_data = stl_located(create_dir_path(frame_dir2))

        for frme2 in sorted(dict_of_path_to_frames[frame_dir2]):
            print(create_dir_path(frame_dir2, frme2))


            ticker += 1


            # FEMUR
            vox_data_mat[2 * ticker] = temp_vox_data[0].astype(save_as_type)  # if len(temp_vox_data[0]) != 1 else print('\n' * 50, 'LOOK HERE', '\nticker = ', 2 * ticker, '\n' * 50)


            # TIBIA
            vox_data_mat[2 * ticker + 1] = temp_vox_data[1].astype(save_as_type)  # if len(temp_vox_data[0]) != 1 else print('\n' * 50, 'LOOK HERE', '\nticker = ', 2 * ticker + 1, '\n' * 50)

    vox_data_mat = np.array(vox_data_mat)

    # vox_dset = vox_file.create_dataset('vox_dset', data=vox_data_mat)

    # vox_file.close()
    try:
        clsd_dset = variable_matrix_storer(vox_data_mat, create_dir_path(path_to_save_compilation, storage_file_name + '.h5py'))
        we_tried = clsd_dset
    except:
        e = sys.exc_info()[0]
        print(e)
        we_tried = 'Well, we tried'

    return we_tried





if __name__ == '__main__':


    top_level_dir = '/Users/johndrago/fluoro/data'

    sys.path.append(os.getcwd())

    # First generate the dictionary of acts, with the corresponding patients who completed those acts
    dict_of_act_pts = generate_dict_of_acts_with_patients()

    # Next create dict of path to frames and then compile a list of frames in that path
    dict_of_frames = generate_dict_path_to_frames(dict_of_act_pts)

    # Generate a list of the paths
    list_of_path_to_frames = list(dict_of_frames.keys())

    # Sort the paths in alphabetical order
    list_of_path_to_frames.sort()

    # path_to_save_compilation = '/Users/johndrago/fluoro/data/compilation'
    path_to_save_compilation = '/Volumes/Seagate/fluoro'



    # Generate variable size voxel and save it to disk. First dimension should be total number of instances we will train on later
    # attempt1 = generate_and_save_mesh_voxel_compilation_matrix(list_of_path_to_frames=list_of_path_to_frames, dict_of_path_to_frames=dict_of_frames, path_to_save_compilation=path_to_save_compilation, storage_file_name='voxels_mark_origin')

    # attempt1 = generate_and_save_mesh_voxel_compilation_matrix(list_of_path_to_frames=list_of_path_to_frames, dict_of_path_to_frames=dict_of_frames, path_to_save_compilation=path_to_save_compilation)


    # -----------------------------------------------------------------

    # dir_file = open(os.path.join(os.getcwd(), 'vox_fluoro_hist_objects.pkl'), 'wb')

    # dir_dict = {}
    # dir_dict['top_level_dir'] = top_level_dir
    # dir_dict['dict_of_frames'] = dict_of_frames
    # dir_dict['list_of_path_to_frames'] = sorted(list_of_path_to_frames)
    # dir_dict['path_to_save_compilation'] = path_to_save_compilation

    # pickle.dump(dir_dict, dir_file)

    # dir_file.close()

    # -----------------------------------------------------------------



    # frames_per_stl_pair_dict = {}

    # indx_tracker = []
    # list_ticker = 0


    # for stl_loc in sorted(list_of_path_to_frames)[:30]:
    #     indx_tracker = []
    #     indx_tracker.append(list_ticker)
    #     indx_tracker.append(list_ticker + 1)
    #     # frames_per_stl_pair_dict[stl_loc[-8:-3] + ' - ' +stl_loc[-2:]] = len(dict_of_frames[stl_loc]) * 2
    #     frames_per_stl_pair_dict[stl_loc[-8:-3] + ' - ' + stl_loc[-2:]] = indx_tracker
    #     list_ticker = list_ticker + len(dict_of_frames[stl_loc]) * 2

    # vox_file = h5py.File('/Users/johndrago/fluoro/data/compilation/voxels_pad.h5py', 'r')
    # vox_init = vox_file['vox_dset']


    # simple_voxel_graph(vox_init[frames_per_stl_pair_dict['CR 08 - Rt'][0]])





    # vox_file.close()





# def voxel_from_array(mesh_vertices, spacing=0.5):
#     '''
#     This function will take in a matrix of the location of mesh vertices. It will then take the vertices and transform them into a binary voxel data set with a 1 located in the bin if a corresponding point is to be found. It will return the voxelized matrix.

#     input:
#         mesh_vertices --> expects np.array of locations of mesh vertices
#         spacing --> the spacing of the voxels in mm
#     output:
#         bin_mat --> a binary voxelized matrix wtih 1's corresponding to points with a corresponding vertex


#     '''
#     mesh_min_vec = np.min(mesh_vertices, axis=0)
#     mesh_min_mat = mesh_vertices - mesh_min_vec
#     range_vec = mesh_vertices.max(axis=0) - mesh_vertices.min(axis=0)
#     bins_vec = np.ceil(range_vec / spacing)
#     bin_mat = np.zeros(bins_vec.astype('int32') + 2)

#     for indx in range(mesh_vertices.shape[0]):
#         # print(int(np.floor(mesh_min_mat[indx, 0] / spacing)))
#         # print(int(np.floor(mesh_min_mat[indx, 1] / spacing)))
#         # print(int(np.floor(mesh_min_mat[indx, 2] / spacing)))

#         # print(type(int(np.floor(mesh_min_mat[indx, 0] / spacing))))
#         # print(type(int(np.floor(mesh_min_mat[indx, 1] / spacing))))
#         # print(type(int(np.floor(mesh_min_mat[indx, 2] / spacing))))


#         bin_mat[int(np.floor(mesh_min_mat[indx, 0] / spacing)):int(np.ceil(mesh_min_mat[indx, 0] / spacing)) + 1, int(np.floor(mesh_min_mat[indx, 1] / spacing)):int(np.ceil(mesh_min_mat[indx, 1] / spacing)) + 1, int(np.floor(mesh_min_mat[indx, 2] / spacing)):int(np.ceil(mesh_min_mat[indx, 2] / spacing)) + 1] = 1

#     return bin_mat.astype('int8')



