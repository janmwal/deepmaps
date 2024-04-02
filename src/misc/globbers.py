import os
from pathlib import Path
import glob

import src.misc.joinfile as misc

def dataset_globber(input_folder, space, center, hemisphere, resolution):
    (space, center, hemisphere, resolution) = [i if i is not None else '*' for i in (space, center, hemisphere, resolution)]
    #print(f'{augmentation_type}/{center}/{hemisphere}/{resolution}um.mat')
    globbing_path = misc.joinfile(input_folder, f'{space}/{center}/{hemisphere}/{resolution}um.mat')
   # print(globbing_path)
    dataset_names = glob.glob(globbing_path)
    if not dataset_names:
            raise ValueError('No dataset found based on given arguments')
    return dataset_names

def vta_globber(input_folder, space, center, hemisphere, resolution, vta_name='VTAs'):
    (space, center, hemisphere, resolution) = [i if i is not None else '*' for i in (space, center, hemisphere, resolution)]

    dataset_names = glob.glob(misc.joinfile(input_folder, f'{space}/{center}/{hemisphere}/{vta_name}/{resolution}um.npz'))
    #print(*dataset_names)
    if not dataset_names:
            raise ValueError('No dataset found based on given arguments')
    return dataset_names
'''
def clean_globber(input_folder, augmentation_type, center, hemisphere, resolution, extension):

    dataset_path = joinfile(input_folder, augmentation_type)
    if not Path(dataset_path).is_dir():
        raise OSError(f'Path {dataset_path} does not exist')

    if not center:
        # if center is not specified, do all the centers (with merged)
        center_names = [f.name for f in os.scandir(dataset_path) if f.is_dir()]
        print('Centers available : ', center_names)
    else:
        if not Path(joinfile(dataset_path, center)).is_dir():
            raise OSError('Center not found in folder')
        center_names = [center]

    # containes the name of the datasets over which we will iterate
    dataset_names = []
    
    if not resolution:
        # if resolution is not specified, we compute all the datasets from the folder   
        for center in center_names:
            # return all files as a list
            for file in os.listdir(joinfile(dataset_path, f'{center}/VTAs')):
                # check the files which are end with specific extension
                if file.endswith(".npz"):
                    # print path name of selected files
                    dataset_names.append(joinfile(f'{dataset_path}/{center}/VTAs',file))
    else:
        # if resolution is specified we only compute this dataset
        dataset_names = joinfile(dataset_path, f'{center}/VTAs/{resolution}um.npz')
        path = Path(dataset_names)
        if not path.is_file():
            raise ValueError('Resolution not found in folder')
        # has to be an iterable for the for loop to work
        dataset_names = [dataset_names]
    
    

def plane_projection_globber(dataset_path, center : str = None, resolution : str = None):
    dataset_names = []
    if not center:
        # if center is not specified, do all the centers (with merged)
        center_names = [f.name for f in os.scandir(dataset_path) if f.is_dir()]
        print('Center names : ', *center_names)
    else:
        if not Path(misc.joinfile(dataset_path, center)).is_dir():
            raise ValueError('Center not found in folder')
        center_names = [center]
    
    if not resolution:
        # if resolution is not specified, we compute all the datasets from the folder   
        for center in center_names:
            # return all files as a list
            for file in os.listdir(misc.joinfile(dataset_path, f'{center}/VTAs')):
                # check the files which are end with specific extension
                if file.endswith(".npz"):
                    # print path name of selected files
                    dataset_names.append(misc.joinfile(f'{dataset_path}/{center}/VTAs',file))
    else:
        # if resolution is specified we only compute this dataset
        for center in center_names:
            dataset_names.append(misc.joinfile(dataset_path, f'{center}/VTAs/{resolution}um.npz'))
            if not Path(dataset_names[-1]).is_file():
                raise ValueError('Resolution not found in folder')
            # has to be an iterable for the for loop to work

    return dataset_names

def dataset_globber(dataset_path, center : str = None, resolution : str = None):
    dataset_names = []
    if not center:
        # if center is not specified, do all the centers (with merged)
        center_names = [f.name for f in os.scandir(dataset_path) if f.is_dir()]
        print('Centers available : ', *center_names)
    else:
        path = Path(misc.joinfile(dataset_path, center))
        if not path.is_dir():
            raise ValueError('Center not found in folder')
        center_names = [center]
    
    if not resolution:
        # if resolution is not specified, we compute all the datasets from the folder
        dataset_names = []
        # return all files as a list
        for center_ in center_names:
            for file in os.listdir(misc.joinfile(dataset_path, f'{center_}')):
                # check the files which are end with specific extension
                if file.endswith(".mat"):
                    # print path name of selected files
                    dataset_names.append(misc.joinfile(f'{dataset_path}/{center_}',file))
    else:
        # if resolution is specified we only compute this dataset
        dataset_names = misc.joinfile(dataset_path, f'{center}/both_hemisphere/{resolution}um_both.mat')
        path = Path(dataset_names)
        if not path.is_file():
            raise ValueError('Resolution not found in folder')
        dataset_names = [dataset_names]
    
    return dataset_names

def shape_features_globber(dataset_path, center : str = None, resolution : str = None):
    if not center:
        # if center is not specified, do all the centers (with merged)
        center_names = [f.name for f in os.scandir(dataset_path) if f.is_dir()]
        print(center_names)
    else:
        path = Path(misc.joinfile(dataset_path, center))
        if not path.is_dir():
            raise ValueError('Center not found in folder')
        center_names = [center]

    # containes the name of the datasets over which we will iterate
    dataset_names = []
    
    if not resolution:
        # if resolution is not specified, we compute all the datasets from the folder   
        for center in center_names:
            # return all files as a list
            for file in os.listdir(misc.joinfile(dataset_path, f'{center}/VTAs')):
                # check the files which are end with specific extension
                if file.endswith(".npz"):
                    # print path name of selected files
                    dataset_names.append(misc.joinfile(f'{dataset_path}/{center}/VTAs',file))
    else:
        # if resolution is specified we only compute this dataset
        for center in center_names:
            dataset_names.append(misc.joinfile(dataset_path, f'{center}/VTAs/{resolution}um.npz'))
            path = Path(dataset_names[-1])
            print(path)
            if not path.is_file():
                raise ValueError('Resolution not found in folder')
            # has to be an iterable for the for loop to work
    return dataset_names
    '''