import torch
import numpy as np
import os
from torch.utils.data import Dataset
from pathlib import Path
from torchvision import transforms
import pandas as pd
from scipy.ndimage import distance_transform_edt
from tqdm import tqdm
from datetime import datetime
import random

from src.misc.joinfile import joinfile
from dir_const import DATA_PROCESSED_DIR, DATA_RAW_DIR

DIR = os.path.dirname(os.path.realpath(__file__))

class ProjectionDataset(Dataset):

    def __init__(
        self, 
        **kwargs):
        super(ProjectionDataset, self).__init__()
        """
        Initialize the Projection dataset
        Arguments:
            center: str -> the center where the data was created
            binarize_labels: bool -> whether to binarize_labels the labels with a threshold
            threshold: int -> the threshold below which the labels are set to zero
            resolutions: int -> the expected resolution of the dataset
        """
        tweening = kwargs.get('tweening', None)

        noise_factor = kwargs.get('noise_factor', None)

        weighted = kwargs.get('weighted', True)

        tuning = kwargs.get('tuning', None) 
        n_split = kwargs.get('n_split', None)

        if kwargs.get('input_folder_vtas', None) is None:
            input_folder_vtas = DATA_PROCESSED_DIR
            #print("Input folder not specified : using 'data/processed' by default")

        if kwargs.get('input_folder_table', None) is None:
            input_folder_table = DATA_RAW_DIR
        space = kwargs.get('space', None)

        dataset_path = joinfile(input_folder_vtas, space)
        #print(f'\n\n{dataset_path}')
        if not Path(dataset_path).is_dir():
            raise OSError('Space not found in input_folder')
        center = kwargs.get('center', None)
        #centers = [f.name for f in os.scandir(dataset_path) if f.is_dir()]
         # if specified center is not in list, raise AssertionError
        #assert center in centers, f'Center not available ({center}) in {centers}'     

        hemispheres = [f.name for f in os.scandir(joinfile(dataset_path, 'merged')) if f.is_dir()]
        hemisphere = kwargs.get('hemisphere', None)
        assert hemisphere in hemispheres, f'Hemisphere not available ({hemisphere})'

        #normalize_projections = kwargs['normalize_projections']
        norm_type = 'raw' #if not normalize_projections else 'norm'

        # get resolution names from the data/processed/{center}/projections/proj_x
        # (resolutions should be the same for all three planes : x, y, z)
        # example : '1200um.npy' -> 1200
        resolutions = [int(f.name.split('um')[0]) for f in os.scandir(joinfile(
            dataset_path, f'merged/{hemisphere}/VTAs_tweened/'))]

        # if resolution is not in list, raise AssertionError
        resolution = kwargs.get('resolution', None)
        assert resolution in resolutions, f'Resolution {resolution}um not available in {resolutions}'

        # load VTAs and y data
        with np.load(joinfile(dataset_path, f'merged/{hemisphere}/VTAs_tweened/{resolution}um.npz')) as f:
            VTAs = f['arr_0']

        interpolation = kwargs.get('interpolation', None)
        augmentation_type = kwargs.get('augmentation_type', None)
        # excluding _Xsigma from space names for the tables directory

        #table_path = joinfile(input_folder_table, f'tables/{space_table}/{center}/{hemisphere}/{table_name}.csv')
        table_path = f'/media/brainstimmaps/DATA/2009_DeepMaps01/04_Source/01_Development/deepmaps/data/processed/{space}/{center}/{hemisphere}/tables_tweened/{resolution}um.csv'
        # TODO : rethink the table placement in the folders
        df = pd.read_csv(table_path)
            
        assert len(VTAs) == len(df), f'table ({len(df)}) and projections ({len(VTAs)}) not the same length'

        idxs = []

        if center != 'merged':
            try:
                idx = df['center'] == center
                idxs.append(idx)
            except:
                raise ValueError(f"{center} does not exist in the table")
            
        if not tweening:
            try:
                idx = df['tweening'] == False
                idxs.append(idx)
            except:
                raise ValueError(f"No tweening column in table")

        if tuning == True:
            try:
                tuning_idx = df[f'split_{n_split}'] == 'tuning'
            except:
                KeyError(f'split #{n_split} not found in table')
            idxs.append(tuning_idx)
        elif tuning == False:
            try:
                tuning_idx = df[f'split_{n_split}'] == 'testing'
            except:
                KeyError(f'split #{n_split} not found in table')
            idxs.append(tuning_idx)
        else:
            print('Taking all the dataset without tuning/testing filter (! overfitting risk !)')
            raise ValueError('tuning param in the dataset should be a bool')
        
        if augmentation_type == 'augmented_full':
            pass
        elif augmentation_type == 'augmented_part':
            part_idx = df['part'] == 1
            idxs.append(part_idx)
        elif augmentation_type == 'mapping':
            map_idx = df['mapping'] == 1
            idxs.append(map_idx)
            #idx = list(set(map_idx) & set(tuning_idx))
        else:
            raise ValueError(f'{augmentation_type} is not : augmented_full, augmented_part or mapping')

        if idxs:
            inter_idx = combine_boolean_lists(idxs)
            print(len(inter_idx))
            VTAs = VTAs[inter_idx]
            #y = score[inter_idx].to_numpy()
            df = df.iloc[inter_idx]

        assert len(VTAs) == len(df), f"VTAs : {len(VTAs)}, df : {len(df)}"

        if noise_factor is not None:
            VTAs, df = surface_noise_augmentation(VTAs, noise_factor, df, weighted)
        
        if interpolation == 'linear_interp':
            y = df['lin_interp_score'].to_numpy()
        elif interpolation == 'step_interp':
            y = df['step_interp_score'].to_numpy()
        elif interpolation == 'no_interp':
            y = df['mapping_score'].to_numpy()
        else:
            raise ValueError(f'{interpolation} is not : lin_interp, step_interp or no_interp')
        
        projections = project(VTAs)

        self.df = df

        # number of sample is the length of VTAs (or y)
        self.n_samples = y.shape[0]

        label_threshold = kwargs.get('label_threshold', None)
        binarize_labels = False if not label_threshold else True

        # if we want to binarize the labels
        if binarize_labels :
            if not (0 <= label_threshold < 1):
                # if threshold is not between 0 and 1, raise AssertionError
                raise ValueError('Threshold must be between 0 and 1')

            # binarize_labels labels
            y[y < label_threshold] = 0.0
            y[y >= label_threshold] = 1.0

            # set a bool flag to indicate the labels have been binarized
            self.binarized = True
        else:
            self.binarized = False
        # convert numpy array to tensor
        self.x_data = torch.from_numpy(projections).float() # size [n_samples, plane, dim1, dim2]

        # convert numpy array to tensor and add dummy 1 channel dimension
        self.y_data = torch.from_numpy(y) # size [n_samples]
        self.y_data = self.y_data.unsqueeze(1) # size [n_samples, 1]

    def subsample(self, ids):
        self.x_data = self.x_data[ids]
        self.y_data = self.y_data[ids]

    def normalize(self, ids = None):
        if ids is None:
            mean = self.x_data.mean()
            std = self.x_data.std()
            self.x_data = transforms.Normalize(mean, std)(self.x_data)
        else:
            mean = self.x_data[ids].mean()
            std = self.x_data[ids].std()
            self.x_data[ids] = transforms.Normalize(mean, std)(self.x_data[ids])
        

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        """
        Return a tuple containing VTAs, y, index at the given index
        Indicies are converted to ints
        Arguments:
            index: int -> the index at which we return VTAs and y
        """
        #print(index)
        return self.x_data[index], self.y_data[index], int(index)

    # we can call len(dataset) to return the size
    def __len__(self):
        """
        Return the number of samples in the dataset
        """
        return self.n_samples

    def get_y(self):
        """
        Return the labels as a torch tensor
        """
        # TODO : 0.85 is arbitrary
        if self.binarized:
            return self.y_data
        else:
            #print(f'Labels are not binarized, binarizing with 0.85 threshold for the balanced class sampling')
            binarized_labels = (self.y_data>0.85).float()
            #print('nan values:', torch.sum(torch.isnan(binarized_labels)).item())
            return binarized_labels

    def _get_labels(self):
        return self.y_data.numpy().squeeze(1)
    
    def get_df(self):
        return self.df

    def get_activation_shape(self):
        """
        Return the the shape of one sample as list of the tuple [(projection, dim1, dim2)]
        """
        #print (self.x_data.size()[1:])
        return list(self.x_data.size()[1:])

    def get_model_args(self):
        """
        Return kwargs for the model constructor
        """
        model_args = {
            'in_shape' : self.get_activation_shape()
        }
        return model_args
    
    def test_stratified_kfold(self, train_ids, test_ids):
        """
        Print the ratio between binarized labels for train and test
        Arguments:
            train_ids: list/tuple of ints -> the indices of the train stratified k-fold split
            test_ids: list/tuple of ints -> the indices of the test stratified k-fold split
        """
        # if the labels have not been previously binarized at initialization, raise AssertionError
        assert self.binarized, 'Labels have not been binarized'

        # print train and test ratio
        print('Test class ratio:', (torch.sum(self.y_data[test_ids])/len(self.y_data[test_ids])).item())
        print('Train class ratio:', (torch.sum(self.y_data[train_ids])/len(self.y_data[train_ids])).item())

        # print real ratio in the dataset
        y = self.y_data.numpy()
        print('Real ratio : ', np.sum(y)/len(y))

    def export_df(self, path):
        try:
            self.df.to_csv(path, index=False)
        except:
            print(f'Could not save df to {path}')

def project(VTAs):

    projs = [(np.sum(VTAs, axis=i).astype(np.int16)) for i in range(1,4)]

    longest_plane = np.max(list(set([x for projection in projs for x in projection.shape[1:]])))

    projs_pad = []

    for i, proj in enumerate(projs):
        (dim1, dim2) = proj.shape[1:]

        # padding each projection with zeros at the right and bottom edges
        # so that the image begins at the top left corner for all projections
        projs_pad.append(np.pad(
            proj, 
            ((0, 0), (0, longest_plane-dim1),(0, longest_plane-dim2)), 
            mode='constant',
            constant_values=0))

    # swap axis 0 and 1 so the shape is [sample, projection, dim1, dim2]
    projs_pad = np.swapaxes(projs_pad, 0, 1)

    return projs_pad

def combine_indices(idxs):
    if len(idxs) == 0:
        return []

    # Start with the first list as the initial intersection set.
    intersection_set = set(idxs[0])

    # Loop through the remaining lists to update the intersection set.
    for i in range(1, len(idxs)):
        intersection_set.intersection_update(idxs[i])

    # Convert the set back to a list and return it.
    return list(intersection_set)

def combine_boolean_lists(lists):
    if len(lists) == 0:
        return []
    
    # Ensure that all lists are of the same length
    list_length = len(lists[0])
    if any(len(lst) != list_length for lst in lists):
        return "All lists must be of the same length."

    # Perform the AND operation element-wise across all lists
    return [all(tup) for tup in zip(*lists)]

def surface_noise_augmentation(VTAs, noise_factor, df, weighted):
    
    df = df.reset_index(drop=True)
    df['noisy'] = False
    df['original_vta'] = np.nan
    df['added_voxels'] = 0

    for id, VTA in enumerate(VTAs):
        df.loc[id, 'total_voxels'] = np.sum(VTA)

    print(f'Surface noise augmentation x {noise_factor:.2f} factor')

    # if we have more than one center, we will compute center-specific noisy factors
    #
    noisy_volumes = [*VTAs]

    centers = df['center'].unique()

    noise_factors = {}
    new_rows = []
    for center in centers:
        df_center = df[df['center'] == center]
        n_center = len(df_center)
        if weighted:
            noise_factors[center] = ((len(df)*(noise_factor+1))/(len(centers)*n_center)) - 1
        else:
            noise_factors[center] = noise_factor
        n_noisy = int(np.floor(n_center*noise_factors[center]))
        random_indices = df_center.sample(n=n_noisy, replace=True).index.tolist()
        print(f'center : {center}, len df_center : {len(df_center)}, len VTAs : {len(VTAs)}')
        #random_selections = [VTAs[i] for i in random_indices]

        for i in tqdm(random_indices):
            amount = np.random.normal(0.0, 0.05)
            amount = np.clip(amount, 0.0, 0.3)

            noisy_volume = add_surface_noise(VTAs[i], amount)
            noisy_volumes.append(noisy_volume)


            new_row = df.loc[i].copy()
            
            new_row['noisy'] = True
            new_row['mapping'] = 0
            new_row['mapping_score'] = np.nan
            new_row['lin_interp_score'] = np.clip(new_row['lin_interp_score'] + amount, 0, 1)
            new_row['step_interp_score'] = np.clip(new_row['step_interp_score'] + amount, 0, 1)

            new_row['original_vta'] = i
            new_row['added_voxels'] = np.sum(noisy_volume-VTAs[i])
            new_row['total_voxels'] = np.sum(noisy_volume)

            new_rows.append(new_row)

    # Convert the list of rows to a DataFrame
    df_rows = pd.DataFrame(new_rows)

    # Concatenate the two DataFrames
    df = pd.concat([df, df_rows], ignore_index=True)

    df = df.reset_index(drop=True)

    noisy_volumes = np.array(noisy_volumes)

    assert len(noisy_volumes) == len(df), f'# noisy volumes ({len(noisy_volumes)}) not equal to # df ({len(df)})'

    return noisy_volumes, df


def add_surface_noise(volume : np.ndarray, amount : float, seed=None):
    # TODO : if amount is negative, subtract voxels from the surface of the shape
    
    # Setting the seed if provided
    if seed is not None:
        np.random.seed(seed)

    # Calculate the distance transform of the volume
    distance = distance_transform_edt(1 - volume)

    # Extracting the surface where the distance to the volume is one voxel away
    surface = np.where(distance==1, 1, 0)

    # Generate random noise 
    noise = np.random.rand(*volume.shape)

    # Binarizing a given proportion of noisy voxels
    noise = np.where(noise >= amount, 0, 1)

    # Masking to the surface
    noise = noise * surface

    # Adding the noise to the original volume
    noisy_volume = volume + noise

    # Returing the ndarray as 8-bit unsigned integer type
    return noisy_volume.astype(np.uint8)