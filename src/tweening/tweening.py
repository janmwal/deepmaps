# -*- coding: utf-8 -*-
import logging

import numpy as np
import pandas as pd
import argparse
import sys

sys.path.append("/media/brainstimmaps/DATA/2009_DeepMaps01/04_Source/01_Development/deepmaps/")

#import pickle
from pathlib import Path
from tqdm import tqdm
from scipy.ndimage import center_of_mass, shift


from dir_const import DATA_PROCESSED_DIR, DATA_INTERIM_DIR
from src.misc.joinfile import joinfile
from src.misc.globbers import vta_globber


def is_index_reset(df):
    return df.index.equals(pd.RangeIndex(start=0, stop=len(df), step=1))


# TODO : write tweening script with automatised dataset loading
def check_voxel_sum_increase(images):
    """
    Check if the sum of voxel values in each 3D volume in the 
    given array of images is greater than the sum in the previous volume. 
    The array is expected to have shape (n_samples, x_dim, y_dim, z_dim).

    Parameters:
    images (np.ndarray): 4D numpy array containing the 3D image data.

    Returns:
    bool: True if it is NOT the case that the sum of voxel values in 
            each 3D volume is greater than the sum in the previous volume, 
            False otherwise.
    """
    
    # Calculate the sum of voxel values for each 3D volume
    sums = np.sum(images, axis=(1, 2, 3))
    diffs = np.array([sums[i]-sums[i-1] for i in range(1, len(sums))])
    # Compare each sum with the previous one
    for i in range(1, len(sums)):
        if sums[i] < sums[i - 1]:
            print(diffs)
            return True
    return False

def interpolate_sparse_shapes(shape1, shape2, num_frames):
    """
    Interpolate between two sparse, binary 2D numpy arrays over a number of frames.
    """
    
    assert shape1.shape == shape2.shape, "Shapes should have the same dimensions"
    #num_frames = num_frames+2
    #centroids = []

    # Find centroids
    centroid1 = np.array(center_of_mass(shape1))
    centroid2 = np.array(center_of_mass(shape2))
    
    #radius1 = compute_radius(shape1, centroid1)
    #radius2 = compute_radius(shape2, centroid2)
    #print(radius1, radius2)
    # Vector connecting centroids from 1 to 2
    vector = centroid2 - centroid1

    #centroids.append(centroid1)
    #print(centroid1)
    frames = [shape1]  # Start with the initial shape
    for i in range(1, num_frames+1):
        alpha = i / (num_frames+1)  # Adjusted alpha to make the step proportional
        
        # Shift shapes without wrapping using mode='nearest'
        shifted_shape1 = shift(shape1*(1-alpha), alpha * vector, mode='constant', order=1)
        shifted_shape2 = shift(shape2*alpha, -(1-alpha) * vector, mode='constant', order=1)

        #interpolated_radius = np.round(alpha * radius2 + (1-alpha) * radius1)
        #print(interpolated_radius)
        # Combine (OR) the shifted shapes to generate the frame
        frame = shifted_shape1+shifted_shape2

        #new_centroid = alpha * centroid2 + (1-alpha) * centroid1
        #centroids.append(new_centroid)
        #print(new_centroid)
        #mask = within_radius(shape1.shape, new_centroid, interpolated_radius)

        frame = np.heaviside((frame-0.5), 0.0)

        frames.append(frame)
        #frames.append(binary_fill_holes(frame))
    #print(centroid2)
    frames.append(shape2)  # End with the final shape
    #centroids.append(centroid2)
    #frames.pop(1)
    #frames.pop(-2)
    return np.array(frames).astype(np.uint8)#, centroids

def tweening(args): # main func
    """ 
    """
    logger = logging.getLogger(__name__)
    logger.info('projecting volumetric VTAS into 3 orthogonal planes')

    # get args into variables
    input_folder = args.input_folder
    output_folder = args.output_folder

    hemisphere = args.hemisphere
    space = args.space
    center = args.center
    resolution = args.resolution


    if input_folder is None:
        input_folder = DATA_INTERIM_DIR

    if output_folder is None:
        output_folder = DATA_PROCESSED_DIR

    dataset_names = vta_globber(input_folder, space, center, hemisphere, resolution)

    print('Dataset names :')
    for dataset_name in dataset_names:
        print(dataset_name)

    for dataset_name in tqdm(dataset_names, colour = 'blue', 
                                            desc = 'Datasets', 
                                            position = 0,
                                            leave = False,):
        # -- Loading VTAs -----------------------------------------------------

        with np.load(dataset_name) as f:
            VTAs = f['arr_0']

        df = pd.read_csv('/media/brainstimmaps/DATA/2009_DeepMaps01/04_Source/01_Development/deepmaps/data/raw/tables/stn_space/merged/flipped/table.csv')

        assert len(df) == len(VTAs), "VTAs and table not the same length"

        if not is_index_reset(df):
            raise ValueError(f'table is not indexed continuously')
        
        newVTAs = []

        # Group by 'patientID' and 'contactID'
        grouped = df.groupby(['patient', 'contact'])

        new_rows = []

        # Iterating through the groups
        for name, group in tqdm(grouped):
            # Iterating through the rows of the group pairwise
            for i in range(len(group) - 1):
                index1 = group.index[i]
                index2 = group.index[i + 1]

                # Copy the metadata from the original df
                base_row = df.loc[index1].copy()
                sup_row = df.loc[index2].copy()

                #print(f'BASE : {base_row["lin_interp_score"]}  ', end='')

                #print(f"Group {name}: Indices -> {index1}, {index2}")
                VTA1 = VTAs[index1]
                VTA2 = VTAs[index2]

                frames = interpolate_sparse_shapes(VTA1, VTA2, 4)

                #if check_voxel_sum_increase(frames):
                #    print(f'{name} {i*0.5}')
                if i == 0:
                    base_row['tweening'] = False
                    new_rows.append(base_row)
                    
                for j in range(1, 5):  # Assuming 4 interpolated volumes
                    new_row = base_row.copy()

                    new_amplitude = base_row['amplitude'] + 0.1 * j
                    new_row['amplitude'] = new_amplitude

                    lin_interp_score = (j * (sup_row['lin_interp_score'] - base_row['lin_interp_score'])/5.0) + base_row['lin_interp_score']
                    #print(f'{lin_interp_score} ', end='')
                    new_row['lin_interp_score'] = lin_interp_score

                    new_row['tweening'] = True

                    new_row['mapping'] = 0

                    new_row['mapping_score'] = np.nan

                    if base_row['part'] == 1 and sup_row['part'] == 0:
                        new_row['part'] = 0

                    new_rows.append(new_row)

                #print(f'SUP : {sup_row["lin_interp_score"]}')   
                # Append the row corresponding to index2 (the "next" volume in the sequence)
                last_row = df.loc[index2].copy()
                last_row['tweening'] = False
                new_rows.append(last_row)

                newVTAs.extend(frames[:-1])
            newVTAs.append(VTA2)


        # verified : all voxels are [0, 1] and integers (binary image)
        # so conversion to unsigned int8 is legit
        newVTAs = np.array(newVTAs).astype(np.uint8)

        # Create a new DataFrame from the list of new rows
        new_df = pd.DataFrame(new_rows)

        # Reset the index of the new DataFrame
        new_df.reset_index(drop=True, inplace=True)

        # Validate the length
        assert len(new_df) == len(newVTAs), "newVTAs and new table not the same length"
        
        # getting resolution and center from file name
        # example : 'data/processed/stn_space_1sigma/merged/flipped/VTAs/250um.npz'
        resolution = dataset_name.split('/')[-1].split('um')[0]
        center = dataset_name.split('/')[-4]
        hemisphere = dataset_name.split('/')[-3]
        space = dataset_name.split('/')[-5]


        # --- SAVING TO FILE --------------------------------------------------

        path = joinfile(output_folder, f'{space}/{center}/{hemisphere}/VTAs_tweened/{resolution}um.npz')
        Path(path).parents[0].mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, newVTAs)

        path = joinfile(output_folder, f'{space}/{center}/{hemisphere}/tables_tweened/{resolution}um.csv')
        Path(path).parents[0].mkdir(parents=True, exist_ok=True)
        new_df.to_csv(path, index=False)


    # --- END OF SCRIPT -------------------------------------------------------         
    
if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    parser = argparse.ArgumentParser(
        description="Creating the dataset.")

    # mandatory arguments
    parser.add_argument("--input_folder", type=str, 
                        help="Path to the raw datasets folder")
    parser.add_argument("--output_folder", type=str,
                        help="path relative to the project folder of the csv "
                             "table containing stimulation parameters")

    # optional arguments
    parser.add_argument("--space", type=str, #default='stn_space_3sigma_subtracted', 
                        help="Type of VTA containing space : stn_space_Xsigma, common_space, n_space")
    parser.add_argument("--center", type=str, default='merged',
                        help="If this is flagged, only this specific center"
                             "will be processed.")
    parser.add_argument("--hemisphere", type=str, default='flipped', 
                        help="Side of dataset : (both, right_only, left_hemisphere, flipped)")
    parser.add_argument("--resolution", type=int,
                        help="If this is flagged, only this specific resolution "
                             "will be processed.")
    args = parser.parse_args()
    tweening(args)
