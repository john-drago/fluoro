'''
This file will allow us to roughly graph voxel data for visualization using mayavi.
'''

import mayavi.mlab as mlab
# from data_organization import extract_stl_femur_tib
import numpy as np
import h5py
import h5py_multidimensional_array


# path_to_dir = '/Users/johndrago/fluoro/data/Gait Updated/CR 01/Lt'

# fib_tib_pair = extract_stl_femur_tib(path_to_dir)
# fib_bin = fib_tib_pair[0]
# tib_bin = fib_tib_pair[1]


def simple_voxel_graph(bin_voxel):
    '''
    Function will simply take in a binary 3D voxel data set and will plot 3D scatter of the vertices for visualization.
    '''

    if len(bin_voxel.shape) != 3:
        assert ValueError('Need 3D voxel dataset')

    number_of_points_vertices = np.count_nonzero(bin_voxel == 1)

    vertex_matrix = np.zeros((number_of_points_vertices, 3))
    ticker = -1

    for x in range(bin_voxel.shape[0]):
        for y in range(bin_voxel.shape[1]):
            for z in range(bin_voxel.shape[2]):
                if bin_voxel[x, y, z]:
                    ticker = ticker + 1
                    vertex_matrix[ticker] = np.array([x, y, z])

    figure_vox = mlab.figure()
    # vox_verts = mlab.points3d(vertex_matrix[:, 0], vertex_matrix[:, 1], vertex_matrix[:, 2], color=(0.2, 0.2, 0.7), opacity=0.45, mode='point')
    vox_verts = mlab.points3d(vertex_matrix[:, 0], vertex_matrix[:, 1], vertex_matrix[:, 2], color=(0.2, 0.2, 0.7), opacity=0.45, mode='sphere', scale_factor=0.25)

    return figure_vox, vox_verts





if __name__ == '__main__':
    # For testing voxel_graph, see below and uncomment:
    # path_to_dir = '/Users/johndrago/fluoro/data/Gait Updated/CR 01/Lt'
    # fib_tib_pair = extract_stl_femur_tib(path_to_dir)
    # fib_bin = fib_tib_pair[0]
    # tib_bin = fib_tib_pair[1]
    # simple_voxel_graph(fib_bin)

    random_numb = np.random.randint(0, 6364)

    # vox_data = h5py_multidimensional_array.variable_matrix_loader('/Users/johndrago/fluoro/data/compilation/voxels.h5py', 'vox_dset', [random_numb, random_numb + 1])

    vox_file = h5py.File('/Users/johndrago/fluoro/data/compilation/voxels_pad.h5py', 'r')
    vox_init = vox_file['vox_dset']

    # vox_data = h5py_multidimensional_array.variable_matrix_loader('/Users/johndrago/fluoro/data/compilation/voxels.h5py', 'vox_dset', [random_numb, random_numb + 1])

    simple_voxel_graph(vox_init[random_numb])

    vox_file.close()
