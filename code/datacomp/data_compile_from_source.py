
'''
The purpose of this file is to organize the matched frames by compiling: (1) two .png fluoroscopic images and (2) the .mat file
with the results of the matching.

To accomplish this, first adjust the following variables at the top of the file:
 - new_dir_name
 - activity_to_copy
 - dir_parse_list
 - replacement_laterality
 - wanted_files_specific
 - wanted_files_base_CAD

Next, place this file into the "CR TKA Matching" file, and the run the "master_file_mover" function included in this file.

Following this, run the "did_all_files_transfer" function to determine if the files transferred.

'''

import os
import shutil

parent_dir = os.getcwd()

new_dir_name = 'compiled_matching_data'

activity_to_copy = [
    'Gait Updated',
    'Kneeling',
    'Step-Up',
    'STS'
]

# Files to parse through to extract our matching data
dir_parse_list = [
    'CR 01',
    'CR 02',
    'CR 03',
    'CR 04',
    'CR 05',
    'CR 06',
    'CR 07',
    'CR 08',
    'CR 09',
    'CR 10',
    'CR 11',
    'CR 12',
    'CR 13',
    'CR 14',
    'CR 15'
]

replacement_laterality = [
    'Rt',
    'Lt'
]

wanted_files_specific = [

]


wanted_files_base_CAD = [
    'LFemur.stl',
    'LTibia.stl',
    'RFemur.stl',
    'RTibia.stl'
]

cali_base = 'reg2fl'

###
# Let's first make a function that can identify the files that we are going to want to copy
# Specifically the files that have been matched and now end in ".mat"


def files_that_end_in_type_identifier(file_type, exclusion='t.mat', *args):
    '''
    Function returns a list of all files that end in file_type
    '''
    # print('\n')
    # print('Finding all files that end in: ' + file_type)
    # print('\n')
    dir_list_of_files = os.listdir()
    list_of_desired_frames = []
    for file in dir_list_of_files:
        if (file[-len(file_type):] == file_type) and (not file[-(len(file_type) + 1):] == exclusion):
            list_of_desired_frames.append(file)
    return list_of_desired_frames

###
# Next we are going to make a function that can identify what the specific frames that have been matched are and then create a dictionary of the corresponding data from a given frame, organized by frame


def matched_png_finder(desired_frames_list, *args):
    '''
    Takes a list of the desired frames that have output results.
    Function will make a dictionary with each frame, as a four digit number, as the key and the value as the three associated file names that share the same frame number
    '''
    frames_dict = {}
    dir_list_of_files = os.listdir()
    stereotyped_ending = '_results.mat'
    for matched_frame in desired_frames_list:
        list_of_files = []
        # print('\nmatched_frame: ', matched_frame)
        for assoc_file in dir_list_of_files:
            # print('same base: ', (assoc_file[0:(len(matched_frame) - len(stereotyped_ending))] == matched_frame[:-len(stereotyped_ending)]))
            # print('different base: ', (assoc_file[0:(len(matched_frame) - len(stereotyped_ending))] == matched_frame[0:1] + '2' + matched_frame[2:-len(stereotyped_ending)]))
            same_base = (assoc_file[0:(len(matched_frame) - len(stereotyped_ending))] == matched_frame[:-len(stereotyped_ending)])
            diff_base = (assoc_file[0:(len(matched_frame) - len(stereotyped_ending))] == matched_frame[0:1] + '2' + matched_frame[2:-len(stereotyped_ending)])
            if same_base or diff_base:
                # print('assoc_file:', assoc_file)
                list_of_files.append(assoc_file)

        # print('list_of_files ', list_of_files)
        frames_dict[matched_frame[(-len(stereotyped_ending) - 4):-len(stereotyped_ending)]] = list_of_files
    return frames_dict


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



def file_copier(where_files_located, where_files_moved, files_to_move, *args):
    # directory_maker(new_dir_name)
    current_dir = os.getcwd()
    copy_path = where_files_moved
    for file in files_to_move:
        os.chdir(where_files_located)
        shutil.copy(file, copy_path)
        os.chdir(current_dir)



def directory_maker(file_name, *args):
    '''
    This function will make a new directory name if the directory name is not already in the current directory
    '''
    cur_dir = os.getcwd()
    print('Current directory:')
    print(cur_dir)

    if file_name not in os.listdir(cur_dir):
        print('Making new directory named: ' + file_name + ' in ' + cur_dir, '\n')
        os.makedirs("./" + file_name, exist_ok=True)


tracking_changes_list = []


def master_file_mover(activity_to_copy=activity_to_copy, dir_parse_list=dir_parse_list, replacement_laterality=replacement_laterality, new_dir_name=new_dir_name, parent_dir=parent_dir):
    '''
    Put this function in "CR-TKA-Matching".

    This function, after specifying the activity, directories to parse through, laterality of the knee, where to put the new data, and where this is, will transfer all of the data to the specificied location. It will organize the data by frame
    '''
    if not os.path.isdir(new_dir_name):
        os.makedirs("./" + new_dir_name, exist_ok=True)

    os.chdir(parent_dir + '/' + new_dir_name)
    for pt in dir_parse_list:
        for side in replacement_laterality:
            for act in activity_to_copy:
                # top_level_dir = os.getcwd()
                # print('\n\ntop_level_dir: ', os.getcwd())
                file_location = create_dir_path(parent_dir, pt, side, act)
                os.chdir(file_location)
                print('Now in: ', file_location)
                # print('os.getcwd(): ', os.getcwd())
                mat_files = files_that_end_in_type_identifier('.mat')
                dict_of_matched_frames = matched_png_finder(mat_files)
                if not (len(dict_of_matched_frames) == 0):
                    global tracking_changes_list2
                    tracking_changes_list.append(create_dir_path(new_dir_name, act, pt, side))
                for frame in dict_of_matched_frames.keys():
                    print('\t--->', pt, side, act, frame)
                    path_to_frame = create_dir_path(parent_dir, new_dir_name, act, pt, side, frame)
                    # print('Path to new frame location: ', path_to_frame)
                    # print('os.getcwd(): ', os.getcwd())
                    os.makedirs(path_to_frame, exist_ok=True)
                    for file in dict_of_matched_frames[frame]:
                        print(file)
                        # print('os.getcwd(): ', os.getcwd())
                        shutil.copy(file, path_to_frame)
                    # print('os.getcwd(): ', os.getcwd())
                    # print('top_level_dir: ', top_level_dir)
                    # os.chdir(top_level_dir)


def did_all_files_transfer(activity_to_copy=activity_to_copy, dir_parse_list=dir_parse_list, replacement_laterality=replacement_laterality, new_dir_name=new_dir_name, parent_dir=parent_dir, number_of_files=3):
    '''
    This function will check to see if for each frame for each side for each patient, there are three files in the child directory, after the data has been copied.

    It will determine if there are three files. If there are not three files, it will return a list of the directories where there are not at least three files.
    '''
    os.chdir(parent_dir + '/' + new_dir_name)
    running_transfer_tracker = []
    for dr in tracking_changes_list:
        patient_and_laterality_dir = create_dir_path(parent_dir, dr)
        os.chdir(patient_and_laterality_dir)
        for frame in os.listdir():
            frame_dir = create_dir_path(patient_and_laterality_dir, frame)
            os.chdir(frame_dir)
            if len(os.listdir()) != number_of_files:
                running_transfer_tracker.append(frame_dir)

    return running_transfer_tracker


def moving_cali_files(activity_to_copy=activity_to_copy, dir_parse_list=dir_parse_list, replacement_laterality=replacement_laterality, new_dir_name=new_dir_name, parent_dir=parent_dir, number_of_files=3, cali_base=cali_base, cali_file='DFIS_Settings.txt'):

    '''
    This function will move the calibration files from the "cali" folder under "CR **", according to the settings contained in the DFIS_Settings.txt document.
    '''

    if not os.path.isdir(new_dir_name):
        os.makedirs("./" + new_dir_name, exist_ok=True)

    os.chdir(parent_dir + '/' + new_dir_name)

    data_rep = '/Users/johndrago/fluoro/data'

    for pt in dir_parse_list:

        for side in replacement_laterality:

            for act in activity_to_copy:

                settings_loc = create_dir_path(parent_dir, pt, side, act)
                file_text_holder = []
                DFIS_file = open(settings_loc + '/' + cali_file)

                print('\n', '\n', 'Now searching: ', settings_loc)

                for line in DFIS_file:

                    file_text_holder.append(line.rstrip())   # rstrip() removes the '\n' from the end of the line

                    if line.rstrip().lower()[0:6] == cali_base:

                        try:

                            # ********** This section of the code attempts to copy the file over, if the file exists (try, except loop)

                            new_dir_cali_drop = create_dir_path(parent_dir, new_dir_name, pt, act, side)


                            os.makedirs(new_dir_cali_drop, exist_ok=True)

                            # print('    Just made file dirs: ', new_dir_cali_drop)

                            shutil.copy(create_dir_path(parent_dir, pt, 'cali', line.rstrip().lower()), new_dir_cali_drop)
                            shutil.copy(create_dir_path(parent_dir, pt, side, act, cali_file), new_dir_cali_drop)

                            # print('    Just copied: ', '\t', '--->', '\t', line.rstrip(), '\t', cali_file)

                            # ********** This section of the code will attempt to move the cali files to the position of the data repository

                            # data_repository_dir = create_dir_path(data_rep, act, pt, side, 'cali')
                            # os.makedirs(data_repository_dir, exist_ok=True)

                            # print('    Just made: ', data_repository_dir)

                            # print('    Going to copy: ', '\n', '\t\t', create_dir_path(parent_dir, pt, 'cali', line.rstrip().lower()), '-->', data_repository_dir)
                            # print('    Going to copy: ', '\n', '\t\t', create_dir_path(parent_dir, pt, side, act, cali_file), '-->', data_repository_dir)

                            # shutil.copy(create_dir_path(parent_dir, pt, 'cali', line.rstrip().lower()), data_repository_dir)
                            # shutil.copy(create_dir_path(parent_dir, pt, side, act, cali_file), data_repository_dir)



                        except (FileNotFoundError):
                            break

                        break








# for act in activity_to_copy:
#     for pt in dir_parse_list:
#         for side in replacement_laterality:
#             patient_and_laterality_dir = create_dir_path(parent_dir, new_dir_name, act, pt, side)
#             os.chdir(patient_and_laterality_dir)
#             for frame in os.listdir():
#                 frame_dir = create_dir_path(patient_and_laterality_dir, frame)
#                 os.chdir(frame_dir)
#                 if len(os.listdir()) != number_of_files:
#                     running_transfer_tracker.append(frame_dir)








# def data_compile_folder_creator(activity_to_copy=activity_to_copy, dir_parse_list=dir_parse_list, replacement_laterality=replacement_laterality, new_dir_name=new_dir_name, parent_dir=parent_dir):
#     pass





# def directory_parser(dir_list, end_destination=None, *vars):
#     cur_dir = os.getcwd()
#     for dir in dir_list:
#         if dir not in cur_dir:
#             print('Directory not in current directory: break')
#             break







# directory_maker(new_dir_name)
