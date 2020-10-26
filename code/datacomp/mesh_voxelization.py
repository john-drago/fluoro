'''
This file will be used to generate voxel data sets from the meshes by voxelizing the vertices.

The high level overview is that we need to supply a list of directory paths to where the stl files are housed. We will then extract the vertices data, and we will create voxels in this file.
'''

import os
from coord_change import Global2Local_Coord
import scipy.io as sio
import skimage
import numpy as np
import trimesh
import pandas as pd
import sys
from scipy import ndimage
import h5py
import pickle



def stl_load_to_vertex_array(path_to_stl, bone):
    '''
    This function will take the path to the frame location according to laterality, and it will return a mesh object loaded according to what laterality and patient has been specified in path.

    input:
        path_to_stl --> the path to specific laterality ~/data/activity/patient/laterality
        bone --> either 'Tibia' or 'Femur' to specify which bone to load

    output:
        mesh --> returns a trimesh mesh object that has been loaded
    '''

    last_path_split = path_to_stl.split(os.sep)[-1]
    new_path = os.path.join(os.path.normpath(path_to_stl[:-3]), 'stl')

    if last_path_split.lower() == 'lt':
        if bone.lower() == 'femur':
            mesh = trimesh.load(os.path.join(new_path, 'LFemur.stl'))
        elif bone.lower() == 'tibia':
            mesh = trimesh.load(os.path.join(new_path, 'LTibia.stl'))

    elif last_path_split.lower() == 'rt':
        if bone.lower() == 'femur':
            mesh = trimesh.load(os.path.join(new_path, 'RFemur.stl'))
        elif bone.lower() == 'tibia':
            mesh = trimesh.load(os.path.join(new_path, 'RTibia.stl'))
    return mesh


def voxel_from_array(mesh_vertices, spacing=0.5):
    '''
    This function will take in a matrix of the location of mesh vertices. It will then take the vertices and transform them into a binary voxel data set with a 1 located in the bin if a corresponding point is to be found. It will return the voxelized matrix.

    input:
        mesh_vertices --> expects np.array of locations of mesh vertices
        spacing --> the spacing of the voxels in mm
    output:
        bin_mat --> a binary voxelized matrix wtih 1's corresponding to points with a corresponding vertex


    '''
    mesh_min_vec = np.min(mesh_vertices, axis=0)
    mesh_min_mat = mesh_vertices - mesh_min_vec
    range_vec = mesh_vertices.max(axis=0) - mesh_vertices.min(axis=0)
    bins_vec = np.ceil(range_vec / spacing)
    bin_mat = np.zeros(bins_vec.astype('int32') + 2)

    for indx in range(mesh_vertices.shape[0]):
        # print(int(np.floor(mesh_min_mat[indx, 0] / spacing)))
        # print(int(np.floor(mesh_min_mat[indx, 1] / spacing)))
        # print(int(np.floor(mesh_min_mat[indx, 2] / spacing)))

        # print(type(int(np.floor(mesh_min_mat[indx, 0] / spacing))))
        # print(type(int(np.floor(mesh_min_mat[indx, 1] / spacing))))
        # print(type(int(np.floor(mesh_min_mat[indx, 2] / spacing))))


        bin_mat[int(np.floor(mesh_min_mat[indx, 0] / spacing)):int(np.ceil(mesh_min_mat[indx, 0] / spacing)) + 1, int(np.floor(mesh_min_mat[indx, 1] / spacing)):int(np.ceil(mesh_min_mat[indx, 1] / spacing)) + 1, int(np.floor(mesh_min_mat[indx, 2] / spacing)):int(np.ceil(mesh_min_mat[indx, 2] / spacing)) + 1] = 1

    return bin_mat.astype('int8')



def extract_stl_to_voxel(mesh_obj, PTS_file, voxelize_dim=0.5):
    '''
    In sum, this function will:

    1) take an STL file loaded as a mesh object and take the PTS file loaded as a pandas object

    2) using the PTS file, determine local coordinate frame and shift STL point cloud to new local coordinate frame

    3) voxelize the vertices of the point cloud to binary, depending on if a vertex would be in the corresponding voxel

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

    LFemur_Z = np.cross(X_vec, Y_vec)

    x_unit = X_vec / np.linalg.norm(X_vec)
    y_unit = Y_vec / np.linalg.norm(Y_vec)
    z_unit = LFemur_Z / np.linalg.norm(LFemur_Z)


    rot_mat = np.array([
        [x_unit[0], y_unit[0], z_unit[0]],
        [x_unit[1], y_unit[1], z_unit[1]],
        [x_unit[2], y_unit[2], z_unit[2]]])

    origin_mesh_local = np.array(PTS_file[0:2, :].mean(axis=0))

    verts_local_coord = Global2Local_Coord(rot_mat, origin_mesh_local, mesh_obj.vertices)

    bin_mat = voxel_from_array(verts_local_coord, spacing=voxelize_dim)

    return bin_mat.astype('int8')







if __name__ == '__main__':

    dir_file = open('vox_fluoro_hist_objects.pkl', 'rb')
    dir_dict = pickle.load(dir_file)
    dir_file.close()




