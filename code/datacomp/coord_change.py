'''
This script is meant to accomplish four things, all about transforming between coordinate systems.

1) Transform from local coordinate system to global coordinate system.
    - func: Local2Global_Coord
2) Transform from globabl coordinate system to local coordinate system.
    - func: Global2Local_Coord
3) If given given a basis of unit vectors, which can describe coordinate change --> be able to produce the three angular rotations (Tait-Bryan angles): phi, theta, psi
    - func: Basis2Angles
4) If given three Tait-Bryan angles: be able to produce the "rotation matrix", comprising the basis matrix for rotation between two coordinate systems.
    - func: Angles2Basis


For reference: https://www.youtube.com/watch?v=meibWcbGqt4
'''
import autograd.numpy as np
from autograd import grad


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

        rot_mat_guess = Angles2Basis([theta, phi, psi])
        error = rot_mat_guess - rot_mat
        epsilon = 0.000009
        error_binary = (error < epsilon)



    assert error_binary.all()
    return [theta, phi, psi]



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

    return rot_mat



def Basis2Angles_GD(rot_mat):
    '''
    function Basis2Angles_GD(rot_mat)

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
    R_y =    [    0       cos(theta)       0    ]
             [ -sin(phi)       0       cos(phi) ]
    - psi
              [  cos(psi)  -sin(psi)    0 ]
    R_z =     [  sin(phi)   cos(psi)    0 ]
              [     0          0        1 ]
    '''

    phi = np.arcsin(-rot_mat[2, 0])
    psi = np.arcsin((rot_mat[1, 0]) / (np.cos(np.arcsin(-rot_mat[2, 0]))))
    theta = np.arcsin((rot_mat[2, 1]) / (np.cos(np.arcsin(-rot_mat[2, 0]))))

    def loss_fn(angle_array):
        loss = (rot_mat[0, 0] - u_x(angle_array))**2 + (rot_mat[1, 0] - u_y(angle_array))**2 + (rot_mat[2, 0] - u_z(angle_array))**2 + (rot_mat[0, 1] - v_x(angle_array))**2 + (rot_mat[1, 1] - v_y(angle_array))**2 + (rot_mat[2, 1] - v_z(angle_array))**2 + (rot_mat[0, 2] - w_x(angle_array))**2 + (rot_mat[1, 2] - w_y(angle_array))**2 + (rot_mat[2, 2] - w_z(angle_array))**2

        return loss

    grad_loss = grad(loss_fn)

    epsilon = 1e-12
    learning_rate = 0.01


    def learning_rate_scheduler(i):
        learning_rate_update = learning_rate  # * (2 / np.sqrt(i))
        return learning_rate_update

    i = 0
    while loss_fn([theta, phi, psi]) > epsilon:
        print('Iteration:\t', i + 1, '\t\tLoss:\t', loss_fn([theta, phi, psi]))
        i = i + 1
        if i < 50:
            theta = theta - learning_rate * grad_loss([theta, phi, psi])[0]
            phi = phi - learning_rate * grad_loss([theta, phi, psi])[1]
            psi = psi - learning_rate * grad_loss([theta, phi, psi])[2]

        else:
            theta = theta - (i / 2) * learning_rate_scheduler(i) * grad_loss([theta, phi, psi])[0]
            phi = phi - (i / 2) * learning_rate_scheduler(i) * grad_loss([theta, phi, psi])[1]
            psi = psi - (i / 2) * learning_rate_scheduler(i) * grad_loss([theta, phi, psi])[2]


    return [theta, phi, psi]


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




# def Basis2Angles(rot_mat):
#     '''
#     function Basis2Angles(rot_mat)

#     This function will take a "rotation matrix", whereby the columns for an orthonormal basis. The "rotation matrix" should describe the axes of the new coordinate system in terms of the global coordinate system. Matrix should be 3x3 and invertible.
#     [ e_1  e_2  e_3 ]

#     Returns a vector of size 3, which containes the following angles in order:
#     '''


#     # phi = np.arcsin(-rot_mat[2, 0])
#     # psi = np.arcsin((rot_mat[1, 0]) / (np.cos(np.arcsin(-rot_mat[2, 0]))))
#     # theta = np.arctan(rot_mat[2, 1] / rot_mat[2, 2])
#     # theta = np.arcsin((rot_mat[2, 1]) / (np.cos(np.arcsin(-rot_mat[2, 0]))))
#     # theta = np.arccos((rot_mat[2, 2]) / (np.cos(np.arcsin(-rot_mat[2, 0]))))


# #     phi = np.arcsin(-rot_mat[2, 0])
# #     psi = np.arcsin((rot_mat[1, 0]) / (np.cos(phi)))
# #     if rot_mat[2, 0] / np.cos(phi) < 0:
# #         psi = pi - psi

# #     theta = np.arcsin((rot_mat[2, 1]) / (np.cos(phi)))
# #     if rot_mat[2, 2] / np.cos(phi) < 0:
# #         theta = pi - theta


# #     rot_mat_guess = Angles2Basis([theta, phi, psi])

# #     error = rot_mat_guess - rot_mat
# #     epsilon = 0.000009

# #     error_binary = (error < epsilon)

# #     if not error_binary.all():
# #         phi = np.pi - phi


#     phi = np.arcsin(-rot_mat[2, 0])
#     psi = np.arcsin((rot_mat[1, 0]) / (np.cos(phi)))
#     if rot_mat[0, 0] / np.cos(phi) < 0:
#         psi = np.pi - psi


#     theta = np.arcsin((rot_mat[2, 1]) / (np.cos(phi)))
#     if rot_mat[2, 2] / np.cos(phi) < 0:
#         theta = np.pi - theta

#     rot_mat_guess = Angles2Basis([theta, phi, psi])

#     error = rot_mat_guess - rot_mat


#     epsilon = 0.000009


#     error_binary = (error < epsilon)

#     if not error_binary.all():
#         phi = np.pi - phi
#         psi = np.arcsin((rot_mat[1, 0]) / (np.cos(phi)))
#         if rot_mat[0, 0] / np.cos(phi) < 0:
#             psi = np.pi - psi


#         theta = np.arcsin((rot_mat[2, 1]) / (np.cos(phi)))
#         # if rot_mat[2, 2] / np.cos(phi) < 0:
#         #     theta = np.pi - theta

#         rot_mat_guess = Angles2Basis([theta, phi, psi])
#         error = rot_mat_guess - rot_mat
#         epsilon = 0.000009
#         error_binary = (error < epsilon)



#     assert error_binary.all()
#     return [theta, phi, psi]


