# -*- coding: utf-8 -*-
import argparse
import logging
import numpy as np
from pathlib import Path

import export_vta_to_vtk as vtk
from misc.joinfile import joinfile
from misc.globbers import vta_globber
from dir_const import DATA_DIR, DATA_PROCESSED_DIR

def main(args):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making data set from raw data')

    input_folder = args.input_folder
    output_folder = args.output_folder
    export_vtk = args.export_vtk

    center = args.center
    augmentation_type = args.augmentation_type
    interpolation_ = args.interpolation
    hemisphere = args.hemisphere  
    resolution = args.resolution

    if input_folder is None:
        input_folder = DATA_PROCESSED_DIR

    if output_folder is None:
        output_folder = DATA_PROCESSED_DIR

    dataset_names = vta_globber(input_folder, center, augmentation_type, hemisphere, resolution)

    for dataset_name in dataset_names :

        # example : 'data/processed/augmented_part/bern/flipped/VTAs/250um.npz'
        resolution = dataset_name.split('/')[-1].split('.')[0][:-2]
        hemisphere = dataset_name.split('/')[-3]
        center = dataset_name.split('/')[-5]
        augmentation_type = dataset_name.split('/')[-4]
        print(f'Center: {center}. Augmentation type : {augmentation_type}. Hemisphere : {hemisphere}. Resolution: {resolution}um. ')
        #print(interpolation_)
        if augmentation_type in ('augmented_part', 'augmented_full'):
            if interpolation_ is None:
                interpolation_list = ('linear_interp', 'step_interp')
            elif interpolation_ == 'linear_interp':
                interpolation_list = ('linear_interp',)
            elif interpolation_ == 'step_interp':
                interpolation_list = ('step_interp',)
            else:
                raise ValueError('Given interpolation does not exist')
        else:
            interpolation_list = ('no_interp',)

        # loading dataset       
        with np.load(dataset_name) as f:
            VTAs = f['arr_0']
        
        # N-image computation
        n_image = np.sum(VTAs, axis=0)

        path = joinfile(output_folder, f'{center}/{augmentation_type}/{hemisphere}/N-images/{resolution}um.npz')
        Path(path).parents[0].mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, n_image)

        if export_vtk :
            n_image_output_filepath = joinfile(output_folder, 
                                    f'{center}/{augmentation_type}/{hemisphere}/N-images/{resolution}um')
            vtk.export(path, n_image_output_filepath, 0)

        for interpolation in interpolation_list:
            labels_path = "/".join(dataset_name.split('/')[:-2]) + f'/labels/{interpolation}/labels.npz'
            with np.load(labels_path) as f:
                y = f['arr_0']

        # Mean-image computation
            mean_image = VTAs.T.dot(y).T
            mean_image = np.divide(
                mean_image,n_image, out=np.zeros_like(mean_image), where=n_image!=0)

            path = joinfile(output_folder, f'{center}/{augmentation_type}/{hemisphere}/Mean_images/{interpolation}/{resolution}um.npz')
            Path(path).parents[0].mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, mean_image)

            if export_vtk :
                mean_image_output_filepath = joinfile(output_folder, 
                                        f'{center}/{augmentation_type}/{hemisphere}/Mean_images/{interpolation}/{resolution}um')
                vtk.export(path, mean_image_output_filepath, 0)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    parser = argparse.ArgumentParser(
        description="Creating the dataset."
    )
    parser.add_argument("--input_folder", type=str,
                        help="Path to the raw datasets folder")
    parser.add_argument("--output_folder", type=str,
                        help="output folder path (default : data/processed")
    parser.add_argument("--export_vtk", action=argparse.BooleanOptionalAction,
                        default=True, 
                        help="boolean whether the n-image and mean-image will "
                             "be exported to a vtk for external 3D visualization")

    parser.add_argument("--center", type=str, 
                        help="Medical center")
    parser.add_argument("--resolution", type=int,
                        help="If this is flagged, only this specific resolution "
                             "will be processed.")
    parser.add_argument("--augmentation_type", type=str,
                        help="Type of dataset : augmented_part, augmented_full, original")
    parser.add_argument("--hemisphere", type=str, default='flipped', 
                        help="Side of dataset : (both_hemisphere, right_hemisphere, left_hemisphere, flipped)")
    parser.add_argument("--interpolation", type=str, 
                        help="Type of interpolation for the labels in augmented (i.e non mapping) datasets : (linear_interp, step_interp)")

    args = parser.parse_args()
    main(args)