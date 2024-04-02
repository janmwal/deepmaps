# -*- coding: utf-8 -*-
import argparse
import logging
import os
#from bleach import clean
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys

print(sys.path)
sys.path.append("/media/brainstimmaps/DATA/2009_DeepMaps01/04_Source/01_Development/deepmaps/")
print(os.getcwd())

from dir_const import DATA_INTERIM_DIR, DATA_PROCESSED_DIR
#from dotenv import find_dotenv, load_dotenv
from src.misc.joinfile import joinfile
from src.misc.globbers import vta_globber
from skimage.measure import regionprops, label #, regionprops
import src.test.clean_test as test

def get_biggest_region(regions):
    #print([region.area for region in regions])
    return np.argmax([region.area for region in regions]) + 1

def clean_vta(VTA, num : int = None, verbose : bool = False):
    clean_VTA = np.zeros_like(VTA)
    # labelling all the regions
    label_img, num_regions = label(VTA, connectivity=3, return_num=True)

    if verbose:
            if num is not None:
                print(f'VTA #{num} : {num_regions} Regions', end = '\t')
            else:
                print(f'{num_regions} Regions', end = '\t')

    # get area of each region
    regions = regionprops(label_img)

    if verbose:
        for i, region in enumerate(regions):
            print(f'[#{i+1} of area {region.area:3d}]   ', end = '')

    # if there is an artefact (more than one region)
    if num_regions > 1:
        # compare areas and get biggest idx
        biggest_region = get_biggest_region(regions)

        if verbose : 
            print(f'\tBiggest region : #{biggest_region}')

        # everything that is not the biggest region is set to zero
        label_img[label_img != biggest_region] = 0

        # by induction, everything that is not a zero now must be the biggest region, thus is set to one
        label_img[label_img != 0] = 1
        # make sure that 
        # assign cleaned vta to new variable
    else:
        if verbose:
            print()
    clean_VTA = label_img
    return clean_VTA
        

def main(args):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('Cleaning VTAs from artefacts.')

    # get args into variables
    input_folder = args.input_folder
    output_folder = args.output_folder
    space = args.space
    center = args.center
    hemisphere = args.hemisphere
    resolution = args.resolution

    if input_folder is None:
        input_folder = DATA_INTERIM_DIR

    if output_folder is None:
        output_folder = DATA_INTERIM_DIR

    dataset_names = vta_globber(input_folder, space, center, hemisphere, resolution, vta_name='VTAs_uncleaned')

    for dataset_name in dataset_names:
        print(dataset_name)
        # loading dataset
        with np.load(dataset_name) as f:
            VTAs = f['arr_0']
        
        # getting resolution and center from file name
        # example : 'data/interim/stn_space_2sigma/merged/flipped/VTAs/250um.npz'
        resolution = dataset_name.split('/')[-1].split('um')[0]
        center = dataset_name.split('/')[-4]
        hemisphere = dataset_name.split('/')[-3]
        space = dataset_name.split('/')[-5]
        print(f'Space : {space}. Center : {center}. Hemisphere : {hemisphere}. Resolution : {resolution}um.')

        clean_VTAs = np.zeros_like(VTAs)
        for i, VTA in enumerate(VTAs):
            clean_VTAs[i] = clean_vta(VTA, i, verbose=False)

        test.check_cleaned(clean_VTAs)
        test.log_deleted_artefacts(
            VTAs, 
            clean_VTAs, 
            resolution, 
            table='/media/brainstimmaps/DATA/2009_DeepMaps01/04_Source/01_Development/deepmaps/data/raw/tables/stn_space/merged/flipped/table.csv',
            file_name=f'{space}_{center}_{hemisphere}_{resolution}um')

        artefacts = VTAs - clean_VTAs

        # save to output folder (make directory if it doesn't exist yet)
        output_path = joinfile(output_folder, f'{space}/{center}/{hemisphere}/VTAs/{resolution}um.npz')
        Path(output_path).parents[0].mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_path, clean_VTAs)

        output_path = joinfile(output_folder, f'{space}/{center}/{hemisphere}/VTA_artefacts/{resolution}um.npz')
        Path(output_path).parents[0].mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_path, artefacts)


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
    parser.add_argument("--space", type=str, #default='stn_space_3sigma_subtracted', 
                        help="Type of VTA containing space : stn_space_Xsigma, common_space, n_space")
    parser.add_argument("--center", type=str, default='merged', 
                        help="If this is flagged, only this specific center"
                             "will be processed.")
    parser.add_argument("--hemisphere", type=str, default='flipped', 
                        help="Hemipshere : (both, right, left, flipped)")
    parser.add_argument("--resolution", type=int,
                        help="If this is flagged, only this specific resolution "
                             "will be processed.")
       
    args = parser.parse_args()
    main(args)
