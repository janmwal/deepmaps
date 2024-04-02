import torch
import numpy as np
import os
from torch.utils.data import Dataset
from pathlib import Path
from torchvision import transforms

from misc.joinfile import joinfile
from dir_const import DATA_PROCESSED_DIR

DIR = os.path.dirname(os.path.realpath(__file__))

class VTADataset(Dataset):

    def __init__(
        self, 
        **kwargs):
        super(VTADataset, self).__init__()
        """
        Initialize the VTA dataset
        Keyword arguments:
            center: str -> the center where the data was created
            binarize: bool -> whether to binarize the labels with a threshold
            threshold: int -> the threshold below which the labels are set to zero
        """

        if kwargs['input_folder'] is None:
            input_folder = DATA_PROCESSED_DIR
            #print("Input folder not specified : using 'data/processed' by default")
        center = kwargs['center']
        dataset_path = joinfile(input_folder, center)
        #print(center)
        if not Path(dataset_path).is_dir():
            raise OSError('Center not founder in input_folder')
        augmentation_type = kwargs['augmentation_type']
        augmentation_types = [f.name for f in os.scandir(dataset_path) if f.is_dir()]
         # if specified center is not in list, raise AssertionError
        assert augmentation_type in augmentation_types, f'Augmentation type not available ({augmentation_type})'     

        hemispheres = [f.name for f in os.scandir(joinfile(dataset_path, augmentation_type)) if f.is_dir()]
        hemisphere = kwargs['hemisphere']
        assert hemisphere in hemispheres, f'Hemisphere not available ({hemisphere})'

        # get resolution names from the data/processed/{center}/projections/proj_x
        # (resolutions should be the same for all three planes : x, y, z)
        # example : '1200um.npy' -> 1200
        resolutions = [int(f.name[:-6]) for f in os.scandir(joinfile(
            dataset_path, f'{augmentation_type}/{hemisphere}/VTAs'))]

        # if resolution is not in list, raise AssertionError
        resolution = kwargs['resolution']
        assert resolution in resolutions, f'Resolution {resolution}um not available in {resolutions}'

        # load X and y data
        with np.load(joinfile(dataset_path, f'{augmentation_type}/{hemisphere}/VTAs/{resolution}um.npz')) as f:
            X = f['arr_0']
            
        interpolation = kwargs['interpolation']
        # loading the labels as numpy array
        with np.load(joinfile(dataset_path, f'{augmentation_type}/{hemisphere}/labels/{interpolation}/labels.npz')) as f:
            y = f['arr_0']

        # if X and y have not the same length, raise AssertionError
        assert X.shape[0] == y.shape[0], f'X and y do not have the same num of samples'

        # number of sample is the length of X (or y)
        self.n_samples = X.shape[0]

        label_threshold = kwargs['label_threshold']
        binarize_labels = False if not label_threshold else True

        # if we want to binarize the labels
        if binarize_labels :
            if not (0 <= label_threshold < 1):
                # if threshold is not between 0 and 1, raise AssertionError
                raise ValueError('Threshold must be between 0 and 1')

            # binarize labels
            y[y < label_threshold] = 0.0
            y[y >= label_threshold] = 1.0

            # set a bool flag to indicate the labels have been binarized
            self.binarized = True

        # convert numpy array to tensor
        self.x_data = torch.from_numpy(X).float() # size [n_samples, H, W, D]
        # adding dummy 1 channel dimension
        self.x_data = self.x_data.unsqueeze(1) # size [n_samples, 1, H, W, D]

        # convert numpy array to tensor
        self.y_data = torch.from_numpy(y) # size [n_samples]
        # adding dummy 1 channel dimension
        self.y_data = self.y_data.unsqueeze(1) # size [n_samples, 1]

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        """
        Return a tuple containing X and y at the given index
        Arguments:
            index: int -> the index at which we return X and y
        """
        return self.x_data[index], self.y_data[index], index

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
        return self.y_data

    def _get_labels(self):
        return self.y_data.numpy().squeeze(1)

    def get_activation_shape(self):
        return list(self.x_data.size()[1:])

    def get_model_args(self):
        """
        Return the the shape of one VTA as one-element list of the tuple [(x, y, z)]
        """
        model_args = {
            'in_shape' : self.get_activation_shape()
        }
        return model_args

    def normalize(self, ids = None):
        #print(self.x_data.shape)
        #print(self.x_data.dtype)
        if ids is None:
            mean = self.x_data.mean()
            std = self.x_data.std()
            self.x_data = transforms.Normalize(mean, std)(self.x_data)
        else:
            mean = self.x_data[ids].mean()
            std = self.x_data[ids].std()
            self.x_data[ids] = transforms.Normalize(mean, std)(self.x_data[ids])
        
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
