# -*- coding: utf-8 -*-
import logging
import os
from matplotlib.cbook import flatten
import numpy as np
import pandas as pd
import argparse
import sys

sys.path.append("/media/brainstimmaps/DATA/2009_DeepMaps01/04_Source/01_Development/deepmaps/")


#import pickle
from pathlib import Path
from tqdm import tqdm
from sklearn.preprocessing import minmax_scale, maxabs_scale, scale

from dir_const import DATA_PROCESSED_DIR
from src.misc.joinfile import joinfile
from src.misc.globbers import vta_globber

def plane_projection(args): # main func
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
        input_folder = DATA_PROCESSED_DIR

    if output_folder is None:
        output_folder = DATA_PROCESSED_DIR

    dataset_names = vta_globber(input_folder, space, center, hemisphere, resolution, vta_name='VTAs_tweened')

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
        
        # getting resolution and center from file name
        # example : 'data/processed/stn_space_1sigma/merged/flipped/VTAs/250um.npz'
        resolution = dataset_name.split('/')[-1].split('um')[0]
        center = dataset_name.split('/')[-4]
        hemisphere = dataset_name.split('/')[-3]
        space = dataset_name.split('/')[-5]

        # --- PROJECTING VTAS INTO EACH PLANE --------------------------------- 
        projs_raw, projs_maxabs_scale, projs_dimdiv, projs_minmax, projs_meanscale = [], [], [], [], []
        projections = {}
        # shape : [projection, sample, dim1, dim2]
        # divide by max, divide by maxdim, minmax, mean scale
        for i in range(1,4): # 1, 2, 3
            proj = np.sum(VTAs, axis=i)
            proj = proj.astype(np.int16)

            # raw : no scaling
            projs_raw.append(proj)

            #maxabs_scale : 
            proj_maxabs_scale = maxabs_scale(proj.reshape(-1, proj.shape[-1])).reshape(proj.shape)
            projs_maxabs_scale.append(proj_maxabs_scale)

            #dimdiv : divide by length of dim for each projection
            proj_dimdiv = np.divide(proj, VTAs.shape[i])
            projs_dimdiv.append(proj_dimdiv)

            #minmax : divide by feature interval (max-min) so it becomes [0,1]
            proj_minmax = minmax_scale(proj.reshape(-1, proj.shape[-1])).reshape(proj.shape)
            projs_minmax.append(proj_minmax)

            #meanscale
            proj_meanscale = scale(proj.reshape(-1, proj.shape[-1])).reshape(proj.shape)
            projs_meanscale.append(proj_meanscale)

        projections['raw'] = projs_raw
        projections['maxabs_scale'] = projs_maxabs_scale
        projections['dimdiv'] = projs_dimdiv
        projections['minmax'] = projs_minmax
        projections['meanscale'] = projs_meanscale
        #proj_types = projections.keys()
        
        for key, item in projections.items():     
            # --- FLATTENING PROJECTIONS FOR 1D FEATURE VECTOR ----------------
            #print(key)
            flatten_projections = []
            for projection in item:
                #print(projection.shape)
                flatten_projections.append(
                    np.reshape(projection, (VTAs.shape[0],-1)))

            flatten_projections = np.concatenate(
                [proj for proj in flatten_projections], axis=1)       
            # shape : n_samples, ((dim1*dim2)+(dim2*dim3)+(dim1*dim3))

            path = joinfile(output_folder, f'{space}/{center}/{hemisphere}/projections/flatten/{key}/{resolution}um.npy')
            Path(path).parents[0].mkdir(parents=True, exist_ok=True)
            with open(path, 'wb') as f:
                np.save(f, flatten_projections) 

            # --- PADDING AND SWAPING AXIS ------------------------------------
            proj_pad = []
            # get longest dimension
            longest_plane = np.max(list(set([x for projection in item for x in projection.shape[1:]])))

            for i, proj in enumerate(item):
                (dim1, dim2) = proj.shape[1:]

                # padding each projection with zeros at the right and bottom edges
                # so that the image begins at the top left corner for all projections
                proj_pad.append(np.pad(
                    proj, 
                    ((0, 0), (0, longest_plane-dim1),(0, longest_plane-dim2)), 
                    mode='constant',
                    constant_values=0))

            # swap axis 0 and 1 so the shape is [sample, projection, dim1, dim2]
            proj_pad = np.swapaxes(proj_pad, 0, 1)

            # --- SAVING TO FILE --------------------------------------------------

            path = joinfile(output_folder, f'{space}/{center}/{hemisphere}/projections/3-channels/{key}/{resolution}um.npy')
            Path(path).parents[0].mkdir(parents=True, exist_ok=True)
            with open(path, 'wb') as f:
                np.save(f, proj_pad)  


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
    parser.add_argument("--space", type=str,
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
    plane_projection(args)
