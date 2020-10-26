'''
This file is developed to help deal with saving multidimensional arrays (4-D) that have variable last three-dimensions. This file is meant to help store variable voxel data sets with a variety of sizes.
'''
import numpy as np
import h5py
import math
import os
import tempfile



def matrix_unwrapper_3d(matrix):
    '''
    This function will unwrap a 3-dimensional array in a consistent way and then create a new 1-D array with all of the data points.

    input:
        matrix --> numpy array 3D

    output:
        array --> unravels in order: along columns, along rows, and then along z_dir
        shape --> of original

    '''

    mat_shape = matrix.shape

    z_dir = mat_shape[0]
    rows = mat_shape[1]
    columns = mat_shape[2]

    # init_array = np.zeros((z_dir * rows * columns))
    init_array = np.reshape(matrix, (z_dir * rows * columns))

    # ticker = -1

    # for z in range(z_dir):
    #     for r in range(rows):
    #         for c in range(columns):
    #             ticker += 1

    #             init_array[ticker] = matrix[z, r, c]


    return [init_array, mat_shape]


def matrix_rewrapper_3d(array, shape):
    '''
    This function will take a 1-d array and rewrap it in a consistent way to reform voxel data:

    input:
        array --> numpy array 1D
        shape --> shape of new matrix as a tuple

    output:
        matrix --> matrix with following dimensions: (z_dir, rows, columns)
    '''
    # init_mat = np.random.rand(shape[0], shape[1], shape[2])

    # z_dir = shape[0]
    # rows = shape[1]
    # columns = shape[2]

    init_mat = np.reshape(array, (shape[0], shape[1], shape[2]))

    # ticker = -1

    # for z in range(z_dir):
    #     for r in range(rows):
    #         for c in range(columns):

    #             ticker += 1
    #             init_mat[z, r, c] = array[ticker]

    return init_mat


def variable_matrix_storer(vox_data_var, file_name_w_path, dset_name='vox_dset', save_as_type='uint8'):
    '''
    This function will create a new h5py file and dataset when given a 4D dataset containing several instances of 3D voxel data that is variable length.

    It will first reshape the individual 3D arrays into 1D arrays with matrix_unwrapper_3d. It will then store each matrix as a 1D array in a numpy array under the name dset_name.

    An attirbute of dset_name will be a matrix of shapes, corresponding the original shape of the matrices before they were reshaped. This will be important for when the 3D matrices are regenerated.


    input:
        vox_data_var --> this function will take the 4D matrix as an argument, can handle 4D matrices of voxel data sets, where the last three-dimensions are variable, expects numpy array of 3D numpy arrays
        file_name_w_path --> takes string argument describing the path and filename (ending in .h5py) where the dataset is to be stored
        dset_name --> the name of the dataset for indexing purposes

    output:
        vox_dset --> will return the voxel dataset from the closed file. In the process, the file will be created according to file_name_w_path, with a corresponding data set (dset_name), with 'shapes' attribute

    '''
    num_of_inst = len(vox_data_var)

    shapes_matrix = np.zeros((num_of_inst, 3))

    array_list = [0] * num_of_inst

    for inst in range(num_of_inst):

        array_list[inst], shapes_matrix[inst] = matrix_unwrapper_3d(vox_data_var[inst])

    array_list = np.array(array_list)

    vox_file = h5py.File(file_name_w_path, 'w')

    # Need to create a special datatype in h5py because of the "ragged" nature of the variable length input

    var_dtype = h5py.special_dtype(vlen=save_as_type)

    vox_dset = vox_file.create_dataset(dset_name, data=array_list, dtype=var_dtype, compression='gzip')

    shapes_matrix = shapes_matrix.astype('uint16')


    # creating an attriubte for the data set so we will be able to regenerate shapes later on
    vox_dset.attrs['shapes'] = shapes_matrix

    vox_file.close()

    return vox_dset


def max_shape_determination(vox_mat):
    '''
    Function to determine the maximum length across various 3D voxel datasets contained in a 4D matrix.

    input:
        vox_mat --> unpadded 4D matrix, consisting of a 1D numpy array consisting of 3D numpy arrays (varying size)

    output:
        max_shape_vector --> list of length 3 describing maximum size amongst the three dimensions
    '''
    max_shape_vector = [0, 0, 0]

    # print('\n\nMax Shape Vector: ', max_shape_vector, '\n\n', type(max_shape_vector))
    # print('\n\n')


    # print('\n\nVox Mat Shape: ', vox_mat.shape, '\n\n', type(vox_mat.shape))
    # print('\n\n')

    for item in range(vox_mat.shape[0]):
        # print('Item: \t', item)

        # print('\n', max_shape_vector, '\n')
        # print('\n')

        # print('\n\n\n', 'vox_mat[item].shape', vox_mat[item].shape)
        if vox_mat[item].shape[0] > max_shape_vector[0]:
            max_shape_vector[0] = vox_mat[item].shape[0]

        if vox_mat[item].shape[1] > max_shape_vector[1]:
            max_shape_vector[1] = vox_mat[item].shape[1]

        if vox_mat[item].shape[2] > max_shape_vector[2]:
            max_shape_vector[2] = vox_mat[item].shape[2]
    return max_shape_vector


def variable_matrix_padder(vox_mat, known_max_shape_vector=None):
    '''
    This function will take a 4D matrix containing many 3D voxel datasets of varying size. It will find the longest size amongst all dimensions and it will return a new 4D matrix with padded 3D voxel datasets all of the same size.

    input:
        vox_mat --> unpadded 4D matrix, consisting of a 1D numpy array consisting of 3D numpy arrays (varying size)

    output:
        vox_mat_pad --> padded 4D numpy array with padded 3D voxel datasets all of the same size
    '''

    # print('\n\nWithin variable matrix padder: \n')
    # print('Vox_mat Shape:\t', vox_mat.shape)

    # for i in range(vox_mat.shape[0]):
    #     print(vox_mat[i].shape)

    if known_max_shape_vector:
        max_shape_vector = known_max_shape_vector
        vox_mat_pad_shape = list(max_shape_vector)
        vox_mat_pad_shape.insert(0, vox_mat.shape[0])
        vox_mat_pad = np.zeros(vox_mat_pad_shape)

    else:
        max_shape_vector = max_shape_determination(vox_mat)

        # print('\n\nMax Shape Vector: ', max_shape_vector, '\n\n', type(max_shape_vector))
        # print('\n\n')

        vox_mat_pad_shape = list(max_shape_vector)
        vox_mat_pad_shape.insert(0, vox_mat.shape[0])
        vox_mat_pad = np.zeros(vox_mat_pad_shape)




    for item in range(vox_mat.shape[0]):
        pad_mat = np.zeros((3, 2))
        # pad_mat = [[0, 0], [0, 0], [0, 0]]

        # print('------------\n\nMatrix shape: \n', vox_mat[item].shape, '\n\n-----------', '\n\n')
        # print('------------\n\nMax Shape Vector: \n', max_shape_vector, '\n\n-----------', '\n\n')
        if vox_mat[item].shape[0] < max_shape_vector[0]:
            pad_mat_0 = max_shape_vector[0] - vox_mat[item].shape[0]
            # print('\n\npad_mat_0:\n', pad_mat_0, '\n\n')
            if pad_mat_0 % 2 == 1:
                pad_mat[0, 0] = pad_mat_0 // 2 + 1
                pad_mat[0, 1] = pad_mat_0 // 2
            else:
                pad_mat[0, :] = pad_mat_0 / 2
                # pad_mat[0][1] = pad_mat_0 / 2

        if vox_mat[item].shape[1] < max_shape_vector[1]:
            pad_mat_1 = max_shape_vector[1] - vox_mat[item].shape[1]
            # print('\n\npad_mat_1:\n', pad_mat_1, '\n\n')
            if pad_mat_1 % 2 == 1:
                pad_mat[1, 0] = pad_mat_1 // 2 + 1
                pad_mat[1, 1] = pad_mat_1 // 2
            else:
                pad_mat[1, :] = pad_mat_1 / 2
                # pad_mat[1][1] = pad_mat_1 / 2

        if vox_mat[item].shape[2] < max_shape_vector[2]:
            pad_mat_2 = max_shape_vector[2] - vox_mat[item].shape[2]
            # print('\n\npad_mat_2:\n', pad_mat_2, '\n\n')
            if pad_mat_2 % 2 == 1:
                pad_mat[2, 0] = pad_mat_2 // 2 + 1
                pad_mat[2, 1] = pad_mat_2 // 2
            else:
                pad_mat[2, :] = pad_mat_2 / 2
                # pad_mat[2][1] = pad_mat_2 / 2

        # print('Pad mat: \n', pad_mat)

        vox_mat_pad[item] = np.pad(vox_mat[item], pad_width=pad_mat.astype(int), mode='constant')

    return vox_mat_pad


def iterative_matrix_padder(vox_mat, size_of_save_obj=50, save_as_type=np.dtype('uint8'), storage_file_path=os.path.expanduser('~/fluoro/data/compilation/'), storage_file_name=None):
    '''
    This function takes a 4D matrix as an argument, and it will use symmetric padding to generate a new 4D matrix that has 3D voxel datasets that are all equivalent sizes.

    This function is specifically useful for when the 3D matrix will not fit in RAM, as it is too big. This function will create a temporary file and then the temporary file will be deleted after the 4D matrix is padded.

    In contrast to sequential iterative matrix padder, this function creates an intermediary matrix that is uploaded to the temporary file, wheres the sequential iterative matrix padder just uploads directly to the storage file. Sequential iterative matrix padder creates a "sub matrix" following download from the temporary file and then uses this sub matrix to do matrix padding before uploading to the storage file.

    input:
        - vox_mat --> the 4D dataset of varying size 3D voxel datasets to complete padding on

    output:
        - store_dset --> h5py dataset stored at storage_file_path. The dset can be accessed after loading the file by calling 'vox_dset'
    '''

    temp_file_base_name = 'tempfile'

    max_shape_vector = max_shape_determination(vox_mat)
    # print(vox_mat.shape)
    vox_mat_pad_shape = list(max_shape_vector)
    vox_mat_pad_shape.insert(0, vox_mat.shape[0])
    # print(vox_mat_pad_shape)
    vox_mat_pad = np.zeros(vox_mat_pad_shape)
    vox_mat_pad = vox_mat_pad.astype(save_as_type)

    dset_dict = {}

    if not storage_file_name:
        storage_file_name = 'voxels_pad'

    store_file = h5py.File(os.path.abspath(os.path.join(storage_file_path, storage_file_name + '.h5py')), 'w')
    store_dset = store_file.create_dataset('vox_dset', data=vox_mat_pad, dtype=save_as_type)

    for item in range(math.ceil(vox_mat_pad_shape[0] / size_of_save_obj)):

        print('\nuploading item:\t', item)
        print('\n')


        # sub_mat = variable_matrix_padder(vox_mat[item * size_of_save_obj:item * size_of_save_obj + size_of_save_obj], known_max_shape_vector=max_shape_vector)
        sub_mat = variable_matrix_padder(vox_mat[0:size_of_save_obj], known_max_shape_vector=max_shape_vector)
        print('sub_mat.shape', sub_mat.shape)
        vox_mat = np.delete(vox_mat, np.s_[:size_of_save_obj:1])
        print('vox_mat.shape', vox_mat.shape)
        dset_dict[temp_file_base_name + '_' + str(item)] = tempfile.TemporaryFile()

        np.save(dset_dict[temp_file_base_name + '_' + str(item)], sub_mat)


    vox_mat = None

    print('\n\n\n')

    for item in range(math.ceil(vox_mat_pad.shape[0] / size_of_save_obj)):

        print('unpacking item:\t', item)

        dset_dict[temp_file_base_name + '_' + str(item)].seek(0)

        temp_data = np.load(dset_dict[temp_file_base_name + '_' + str(item)], allow_pickle=True)

        store_dset[item * size_of_save_obj:item * size_of_save_obj + size_of_save_obj] = temp_data.astype(save_as_type)


    store_file.close()

    vox_mat_pad = None

    return store_dset


def sequential_iterative_matrix_padder(vox_mat, size_of_save_obj=50, save_as_type=np.dtype('uint8'), storage_file_path=os.path.expanduser('~/fluoro/data/compilation/'), storage_file_name=None, compression=None):
    '''
    This function takes a 4D matrix as an argument, and it will use symmetric padding to generate a new 4D matrix that has 3D voxel datasets that are all equivalent sizes.

    This function is specifically useful for when the 3D matrix will not fit in RAM, as it is too big. This function will create a temporary file and then the temporary file will be deleted after the 4D matrix is padded.

    This function is in contrast to iterative matrix padder, which creates an intermediary function prior to saving the data to the storage file, on which the matrix padding funciton is run to get the voxel data sets to the same size.

    *** Would typically want to use sequential_iterative_matrix_padder before iterative_matrix_padder, as it can handle larger files without getting out of memory errors.

    input:
        - vox_mat --> the 4D dataset of varying size 3D voxel datasets to complete padding on

    output:
        - store_dset --> h5py dataset stored at storage_file_path. The dset can be accessed after loading the file by calling 'vox_dset'
    '''

    temp_file_base_name = 'tempfile'

    max_shape_vector = max_shape_determination(vox_mat)
    # print(vox_mat.shape)
    vox_mat_pad_shape = list(max_shape_vector)
    vox_mat_pad_shape.insert(0, vox_mat.shape[0])
    # print(vox_mat_pad_shape)
    # vox_mat_pad = np.zeros(vox_mat_pad_shape)
    # vox_mat_pad = vox_mat_pad.astype(save_as_type)

    dset_dict = {}

    if not storage_file_name:
        storage_file_name = 'voxels_pad'

    store_file = h5py.File(os.path.abspath(os.path.join(storage_file_path, storage_file_name + '.h5py')), 'w')
    if compression:
        store_dset = store_file.create_dataset('vox_dset', shape=tuple(vox_mat_pad_shape), dtype=save_as_type, compression=compression)
    else:
        store_dset = store_file.create_dataset('vox_dset', shape=tuple(vox_mat_pad_shape), dtype=save_as_type)

    for item in range(math.ceil(vox_mat_pad_shape[0] / size_of_save_obj)):

        print('\nuploading item:\t', item)
        print('\n')

        dset_dict[temp_file_base_name + '_' + str(item)] = tempfile.TemporaryFile()

        np.save(dset_dict[temp_file_base_name + '_' + str(item)], vox_mat[0:size_of_save_obj])

        # print('Sub mat\t', vox_mat[0:size_of_save_obj].shape)

        vox_mat = np.delete(vox_mat, np.s_[:size_of_save_obj:1], axis=0)
        print('vox_mat.shape\t', vox_mat.shape)

    vox_mat = None

    print('\n\n\n')

    print(dset_dict.keys())

    print('\n\n\n')

    for item in range(math.ceil(vox_mat_pad_shape[0] / size_of_save_obj)):

        print('unpacking item:\t', item)

        dset_dict[temp_file_base_name + '_' + str(item)].seek(0)

        temp_data = np.load(dset_dict[temp_file_base_name + '_' + str(item)], allow_pickle=True)

        dset_dict[temp_file_base_name + '_' + str(item)].close()

        upload_data = variable_matrix_padder(temp_data, max_shape_vector)

        store_dset[item * size_of_save_obj:item * size_of_save_obj + size_of_save_obj] = upload_data.astype(save_as_type)

        upload_data = None


    store_file.close()

    return store_dset



def variable_matrix_loader(file_name_w_path, dset_name='vox_dset', index_of_instances=[None, None], padding=False, known_max_shape_vector=None):
    '''
    This function will assume that there is a function foo.h5py that has already been created. In foo, there is assumed to be a dataset under the name 'dset_name', which is an input to this function.

    The corresponding data set is assumed to be an array of arrays, where the first index corresponds to the instance number, and the second array is a reshaped 3D voxel data set (into 1D).

    There is also an assumption that the data set has an attribute 'shapes', which describes how the 1D arrays will be reshaped into 3D arrays for processing.

    input:
        - file_name_w_path --> this is a string comprising a path to and the name of the h5py file where the data is stored
        - dset_name --> this is a string describing how the data set in the file should be indexed
        - number of instances --> this is expected to be an indexable object of size two, where the first object is supposed to be where the index of the array of voxel data should start, and the second object is where the index should stop, according to standard Python indexing.
        - index_of_instances --> if nothing, will default to the entire matrix, will otherwise default to array index according to first and last int of list

    output:
        mat_3d_vox
    '''
    vox_file = h5py.File(file_name_w_path, 'r')
    vox_dset = vox_file[dset_name]

    vox_dset_shapes = vox_dset.attrs['shapes']

    if index_of_instances[0] is None and index_of_instances[1] is None:
        vox_dset_subset = vox_dset
        vox_shapes_subset = vox_dset_shapes
    else:
        vox_dset_subset = vox_dset[index_of_instances[0]:index_of_instances[1]]
        vox_shapes_subset = vox_dset_shapes[index_of_instances[0]:index_of_instances[1]]


    mat_3d_vox = [0] * vox_dset_subset.shape[0]

    for inst in range(vox_dset_subset.shape[0]):

        mat_3d_vox[inst] = np.reshape(vox_dset_subset[inst], vox_shapes_subset[inst])


    vox_file.close()

    # return np.array(mat_3d_vox)

    if padding:
        if known_max_shape_vector:
            return variable_matrix_padder(np.array(mat_3d_vox), known_max_shape_vector)
        else:
            return variable_matrix_padder(np.array(mat_3d_vox))

    else:
        return np.array(mat_3d_vox)






if __name__ == '__main__':
    print('\n')

    # import sys
    # sys.path.append(os.path.abspath(os.path.expanduser('~/fluoro/code')))

    # load_path = '~/fluoro/data/compilation/voxels.h5py'
    load_path = '/Volumes/Seagate/fluoro/voxels_mark_origin.h5py'

    vox_mat = variable_matrix_loader(load_path, index_of_instances=[None, None])

    print('loaded max matrix')

    shape_mat = max_shape_determination(vox_mat)

    try1 = sequential_iterative_matrix_padder(vox_mat, size_of_save_obj=100, save_as_type=np.dtype('int8'), storage_file_path='/Volumes/Seagate/fluoro', storage_file_name='voxels_mark_origin_comp', compression='lzf')

    # try1 = sequential_iterative_matrix_padder(vox_mat, size_of_save_obj=100, save_as_type=np.dtype('int8'), storage_file_path='/Volumes/Seagate/fluoro', storage_file_name='voxels_pad', compression=None)

    # try1 = sequential_iterative_matrix_padder(vox_mat, size_of_save_obj=100, save_as_type=np.dtype('int8'), storage_file_path=os.path.expanduser('~/fluoro/data/compilation'), storage_file_name=None, compression='lzf')




    # mat1 = variable_matrix_padder(vox_mat)








    # ------------------------------------------
    # BACKUP ITERATIVE MATRIX PADDER 2: temp file, pad on input





    # def iterative_matrix_padder(vox_mat, size_of_save_obj=750, save_as_type=np.dtype('uint8'), save_as_h5py=False, storage_file_path=os.path.expanduser('~/fluoro/data/compilation/'), storage_file_name=None):
    #     '''
    #     This function takes a 4D matrix as an argument, and it will use symmetric padding to generate a new 4D matrix that has 3D voxel datasets that are all equivalent sizes.

    #     This function is specifically useful for when the 3D matrix will not fit in RAM, as it is too big. This function will create a temporary file and then the temporary file will be deleted after the 4D matrix is padded.

    #     input:
    #         - vox_mat --> the 4D dataset of varying size 3D voxel datasets to complete padding on

    #     output:
    #         - vox_mat_pad --> padded 4D numpy array with padded 3D voxel datasets all of the same size
    #     '''

    #     temp_file_base_name = 'tempfile'

    #     var_dtype = h5py.special_dtype(vlen=save_as_type)

    #     temp_file = h5py.File(os.path.abspath(os.path.join(os.getcwd(), temp_file_base_name + '.h5py')), 'w')


    #     max_shape_vector = max_shape_determination(vox_mat)

    #     # print(vox_mat.shape)
    #     vox_mat_pad_shape = list(max_shape_vector)
    #     vox_mat_pad_shape.insert(0, vox_mat.shape[0])

    #     # print(vox_mat_pad_shape)
    #     vox_mat_pad = np.zeros(vox_mat_pad_shape)

    #     vox_mat_pad = vox_mat_pad.astype(save_as_type)

    #     temp_file.close()

    #     dset_dict = {}


    #     for item in range(math.ceil(vox_mat_pad_shape[0] / size_of_save_obj)):

    #         print('uploading item:\t', item)

    #         temp_file = h5py.File(os.path.abspath(os.path.join(os.getcwd(), temp_file_base_name + '.h5py')), 'a')

    #         sub_mat = vox_mat[item * size_of_save_obj:item * size_of_save_obj + size_of_save_obj]

    #         num_of_inst = len(sub_mat)

    #         shapes_matrix = np.zeros((num_of_inst, 3))

    #         array_list = [0] * num_of_inst

    #         for sub_inst in range(num_of_inst):

    #             array_list[sub_inst], shapes_matrix[sub_inst] = matrix_unwrapper_3d(sub_mat[sub_inst])

    #             # print(array_list[sub_inst].shape)
    #             # print(shapes_matrix[sub_inst])

    #         array_list = np.array(array_list)

    #         dset_dict[temp_file_base_name + '_' + str(item)] = temp_file.create_dataset(temp_file_base_name + '_' + str(item), data=array_list, dtype=var_dtype, compression='gzip')

    #         shapes_matrix = shapes_matrix.astype('uint16')

    #         dset_dict[temp_file_base_name + '_' + str(item)].attrs['shapes'] = shapes_matrix
    #         # print(temp_dset[:].shape)
    #         temp_file.close()

    #     vox_mat = None

    #     print('\n\n\n')

    #     for item in range(math.ceil(vox_mat_pad.shape[0] / size_of_save_obj)):

    #         print('unpacking item:\t', item)

    #         temp_data = variable_matrix_loader(os.path.abspath(os.path.join(os.getcwd(), temp_file_base_name + '.h5py')), dset_name=temp_file_base_name + '_' + str(item), index_of_instances=[None, None], padding=True, known_max_shape_vector=max_shape_vector)
    #         # print('Temp Data:\t', temp_data[:].shape, '\t', type(temp_data[:]), '\t', temp_data[:].dtype)

    #         vox_mat_pad[item * size_of_save_obj:item * size_of_save_obj + size_of_save_obj] = temp_data.astype(save_as_type)


    #     os.remove(os.path.abspath(os.path.join(os.getcwd(), temp_file_base_name + '.h5py')))

    #     vox_mat_pad = vox_mat_pad.astype(save_as_type)

    #     if not storage_file_name:
    #         storage_file_name = 'voxels_pad'

    #     if save_as_h5py:

    #         print('vox_mat_pad.dtype', '\t', vox_mat_pad.dtype)

    #         store_file = h5py.File(os.path.abspath(os.path.join(storage_file_path, storage_file_name + '.h5py')), 'w')

    #         store_dset = store_file.create_dataset('vox_dset', data=vox_mat_pad, dtype=save_as_type, compression='gzip')

    #         store_file.close()

    #         vox_mat_pad = None


    #         return store_dset
    #     else:
    #         return vox_mat_pad








    # ------------------------------------------

    # ------------------------------------------
    # BACKUP ITERATIVE MATRIX PADDER 1: make new datasets in h5py file





    # def iterative_matrix_padder(vox_mat, size_of_save_obj=750, save_as_type=np.dtype('uint8'), save_as_h5py=False, storage_file_path=os.path.expanduser('~/fluoro/data/compilation/'), storage_file_name=None):
    #     '''
    #     This function takes a 4D matrix as an argument, and it will use symmetric padding to generate a new 4D matrix that has 3D voxel datasets that are all equivalent sizes.

    #     This function is specifically useful for when the 3D matrix will not fit in RAM, as it is too big. This function will create a temporary file and then the temporary file will be deleted after the 4D matrix is padded.

    #     input:
    #         - vox_mat --> the 4D dataset of varying size 3D voxel datasets to complete padding on

    #     output:
    #         - vox_mat_pad --> padded 4D numpy array with padded 3D voxel datasets all of the same size
    #     '''

    #     temp_file_base_name = 'tempfile'

    #     var_dtype = h5py.special_dtype(vlen=save_as_type)

    #     temp_file = h5py.File(os.path.abspath(os.path.join(os.getcwd(), temp_file_base_name + '.h5py')), 'w')


    #     max_shape_vector = max_shape_determination(vox_mat)

    #     # print(vox_mat.shape)
    #     vox_mat_pad_shape = list(max_shape_vector)
    #     vox_mat_pad_shape.insert(0, vox_mat.shape[0])

    #     # print(vox_mat_pad_shape)
    #     vox_mat_pad = np.zeros(vox_mat_pad_shape)

    #     vox_mat_pad = vox_mat_pad.astype(save_as_type)

    #     temp_file.close()

    #     dset_dict = {}


    #     for item in range(math.ceil(vox_mat_pad_shape[0] / size_of_save_obj)):

    #         print('uploading item:\t', item)

    #         temp_file = h5py.File(os.path.abspath(os.path.join(os.getcwd(), temp_file_base_name + '.h5py')), 'a')

    #         sub_mat = vox_mat[item * size_of_save_obj:item * size_of_save_obj + size_of_save_obj]

    #         num_of_inst = len(sub_mat)

    #         shapes_matrix = np.zeros((num_of_inst, 3))

    #         array_list = [0] * num_of_inst

    #         for sub_inst in range(num_of_inst):

    #             array_list[sub_inst], shapes_matrix[sub_inst] = matrix_unwrapper_3d(sub_mat[sub_inst])

    #             # print(array_list[sub_inst].shape)
    #             # print(shapes_matrix[sub_inst])

    #         array_list = np.array(array_list)

    #         dset_dict[temp_file_base_name + '_' + str(item)] = temp_file.create_dataset(temp_file_base_name + '_' + str(item), data=array_list, dtype=var_dtype, compression='gzip')

    #         shapes_matrix = shapes_matrix.astype('uint16')

    #         dset_dict[temp_file_base_name + '_' + str(item)].attrs['shapes'] = shapes_matrix
    #         # print(temp_dset[:].shape)
    #         temp_file.close()

    #     vox_mat = None

    #     print('\n\n\n')

    #     for item in range(math.ceil(vox_mat_pad.shape[0] / size_of_save_obj)):

    #         print('unpacking item:\t', item)

    #         temp_data = variable_matrix_loader(os.path.abspath(os.path.join(os.getcwd(), temp_file_base_name + '.h5py')), dset_name=temp_file_base_name + '_' + str(item), index_of_instances=[None, None], padding=True, known_max_shape_vector=max_shape_vector)
    #         # print('Temp Data:\t', temp_data[:].shape, '\t', type(temp_data[:]), '\t', temp_data[:].dtype)

    #         vox_mat_pad[item * size_of_save_obj:item * size_of_save_obj + size_of_save_obj] = temp_data.astype(save_as_type)


    #     os.remove(os.path.abspath(os.path.join(os.getcwd(), temp_file_base_name + '.h5py')))

    #     vox_mat_pad = vox_mat_pad.astype(save_as_type)

    #     if not storage_file_name:
    #         storage_file_name = 'voxels_pad'

    #     if save_as_h5py:

    #         print('vox_mat_pad.dtype', '\t', vox_mat_pad.dtype)

    #         store_file = h5py.File(os.path.abspath(os.path.join(storage_file_path, storage_file_name + '.h5py')), 'w')

    #         store_dset = store_file.create_dataset('vox_dset', data=vox_mat_pad, dtype=save_as_type, compression='gzip')

    #         store_file.close()

    #         vox_mat_pad = None


    #         return store_dset
    #     else:
    #         return vox_mat_pad








    # ------------------------------------------
    # if index_of_instances[0] == None and index_of_instances[1] == None:
    #     instance_index = [0, vox_dset[0]]
    # elif index_of_instances[0] != None and index_of_instances[1] != None:
    #     instance_index = index_of_instances
    # elif index_of_instances[0] == None:
    #     if index_of_instances[1] <0:
    #         instance_index = [0, vox_dset[0]+index_of_instances[1]]
    #     elif index_of_instances[1] >0:
    #         instance_index = [0, index_of_instances[1]]
    #     else:
    #         assert ValueError
    # elif index_of_instances[1] == None:
    #     if index_of_instances[0] <0:
    #         instance_index[]
    #
    #
