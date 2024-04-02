import numpy as np
import src.data.clean_vtas as clean
import skimage.measure as skimage
import warnings
import logging
from datetime import datetime
import pandas as pd
from src.misc.joinfile import joinfile
from dir_const import LOG_DIR


# ----------------------------------------------------------------

def run_clean_tests(verbose : bool):
    print('=== Running tests about cleaning VTAs ===')
    if (test_clean1(verbose) and test_clean2(verbose) and test_clean3(verbose)):
        print('=== All cleaning tests passed successfully ===')
    print()

def test_clean1(verbose : bool):
    if verbose : print('Test clean 1', end='\t')
    VTA = np.array( [   [   [0, 0, 0],
                            [0, 1, 1],
                            [0, 1, 1]],
                        [   [0, 0 ,0],
                            [0, 0, 0],
                            [0, 0, 0]],
                        [   [0, 0, 0],
                            [0, 1, 0], # this should be deleted
                            [0, 0, 0]]])
    clean_VTA = np.array(   [   [   [0, 0, 0],
                                    [0, 1, 1],
                                    [0, 1, 1]],
                                [   [0, 0 ,0],
                                    [0, 0, 0],
                                    [0, 0, 0]],
                                [   [0, 0, 0],
                                    [0, 0, 0],
                                    [0, 0, 0]]])
    processed_VTA = clean.clean_vta(VTA, verbose=verbose)
    #print(cleaned_VTAs)
    if np.array_equal(processed_VTA,clean_VTA):
        if verbose: print('Test clean 1 passed')
        return True
    if verbose: print('Test clean 1 failed')
    return False

def test_clean2(verbose : bool):
    if verbose : print('Test clean 2', end='\t')
    VTA = np.array( [       [   [0, 0, 0],
                                [0, 1, 0], # if area are the same size ...
                                [0, 0, 0]],
                            [   [0, 0 ,0],
                                [0, 0, 0],
                                [0, 0, 0]],
                            [   [0, 0, 0],
                                [0, 1, 0], # ... second area should be deleted
                                [0, 0, 0]]])
    clean_VTA = np.array([  [   [0, 0, 0],
                                [0, 1, 0],
                                [0, 0, 0]],
                            [   [0, 0 ,0],
                                [0, 0, 0],
                                [0, 0, 0]],
                            [   [0, 0, 0],
                                [0, 0, 0],
                                [0, 0, 0]]])
    processed_VTA = clean.clean_vta(VTA, verbose=verbose)
    #print(cleaned_VTAs)
    if np.array_equal(processed_VTA,clean_VTA):
        if verbose: print('Test clean 2 passed')
        return True
    if verbose: print('Test clean 2 failed')
    return False

def test_clean3(verbose : bool):
    if verbose: print('Test clean 3', end='\t')
    VTA = np.array( [   [   [0, 0, 0],
                            [0, 1, 1],
                            [0, 0, 0]],
                        [   [0, 0 ,0],
                            [0, 0, 0],
                            [0, 0, 0]],
                        [   [0, 0, 0],
                            [0, 1, 1], # this should be deleted
                            [0, 1, 1]]])
    clean_VTA = np.array(   [   [   [0, 0, 0],
                                    [0, 0, 0],
                                    [0, 0, 0]],
                                [   [0, 0 ,0],
                                    [0, 0, 0],
                                    [0, 0, 0]],
                                [   [0, 0, 0],
                                    [0, 1, 1],
                                    [0, 1, 1]]])
    processed_VTA = clean.clean_vta(VTA, verbose=verbose)
    #print(cleaned_VTAs)
    if np.array_equal(processed_VTA,clean_VTA):
        if verbose: print('Test clean 3 passed')
        return True
    if verbose: print('Test clean 3 failed')
    return False

# ----------------------------------------------------------------
def run_region_tests(verbose : bool):
    print('=== Running tests about computing the biggest region ===')
    if (test_region1(verbose) and test_region2(verbose) and test_region3(verbose)):
        print('=== All tests about computing the biggest region passed successfully ===')
    else: print('=== One or more tests failed ===')
    print()

def test_region1(verbose : bool):
    if verbose:
        print('Test region 1', end='')
    VTA = np.array( [   [   [0, 0, 0],
                            [0, 1, 1], # biggest region
                            [0, 0, 0]],
                        [   [0, 0 ,0],
                            [0, 0, 0],
                            [0, 0, 0]],
                        [   [0, 0, 0],
                            [0, 1, 0], 
                            [0, 0, 0]]])
    label_img, num_regions = skimage.label(VTA, connectivity=3, return_num=True)
    regions = skimage.regionprops(label_img)
    print(label_img)
    biggest_region = clean.get_biggest_region(regions)
    if verbose:
        print(f'\tbiggest region : {biggest_region}, actual : 1', end ='')
    if biggest_region == 1:
        if verbose:
            print('\tTest biggest region 1 passed')
        return True
    else:
        if verbose:
            print(f'\tTest biggest region 1 failed')
        return False

def test_region2(verbose : bool):
    if verbose:
        print('Test region 2', end='')
    VTA = np.array( [   [   [0, 0, 0],
                            [0, 1, 0],
                            [0, 0, 0]],
                        [   [0, 0 ,0],
                            [0, 0, 0],
                            [0, 0, 0]],
                        [   [0, 0, 0],
                            [0, 1, 1], # biggest region
                            [0, 0, 0]]])
    label_img, num_regions = skimage.label(VTA, connectivity=3, return_num=True)
    regions = skimage.regionprops(label_img)
    print(label_img)
    biggest_region = clean.get_biggest_region(regions)
    if verbose:
        print(f'\tbiggest region : {biggest_region}, actual : 2', end='')
    if biggest_region == 2:
        if verbose:
            print('\tTest biggest region 2 passed')
        return True
    else:
        if verbose:
            print(f'\tTest biggest region 2 failed')
        return False

def test_region3(verbose : bool):
    if verbose:
        print('Test region 3', end='')
    VTA = np.array( [   [   [0, 0, 0],
                            [1, 1, 0], # biggest region
                            [0, 0, 0]],
                        [   [0, 0 ,0],
                            [0, 0, 0],
                            [0, 0, 0]],
                        [   [0, 0, 0],
                            [0, 1, 1], 
                            [0, 0, 0]]])
    label_img, num_regions = skimage.label(VTA, connectivity=3, return_num=True)
    regions = skimage.regionprops(label_img)
    print(label_img)
    biggest_region = clean.get_biggest_region(regions)
    if verbose:
        print(f'\tBiggest region : {biggest_region}, actual : 2', end='')
    if biggest_region == 1:
        if verbose:
            print('\tTest biggest region 3 passed')
        return True
    else:
        if verbose:
            print(f'\tTest biggest region 3 failed')
        return False

# ----------------------------------------------------------------

def run_label_tests(verbose : bool):
    '''
    This function runs a battery of tests for  labelling VTAs.
    This process is very simple because it happens gradually :
    The first region encountered is #1, second #2, etc independently of size.
    Hence the testing of this functionality is straight-forward.
    '''
    print('=== Running tests about labelling regions ===')
    if (test_label1(verbose) and test_label2(verbose) and test_label3(verbose)):
        print('=== All labelling tests passed successfully ===')
    else: print('=== One or more tests failed ===')
    print()

def test_label1(verbose : bool):
    if verbose : print('Test label 1', end='\t')
    VTA = np.array( [   [   [0, 0, 0],
                            [0, 1, 1], # region 1
                            [0, 0, 0]],
                        [   [0, 0 ,0],
                            [0, 0, 0],
                            [0, 0, 0]],
                        [   [0, 0, 0],
                            [0, 1, 1], # region 2
                            [0, 1, 1]]])
    label_VTA = np.array(   [   [   [0, 0, 0],
                                    [0, 1, 1],
                                    [0, 0, 0]],
                                [   [0, 0 ,0],
                                    [0, 0, 0],
                                    [0, 0, 0]],
                                [   [0, 0, 0],
                                    [0, 2, 2],
                                    [0, 2, 2]]])
    processed_VTA, num_regions = skimage.label(VTA, connectivity=3, return_num=True)
    #print(cleaned_VTAs)
    if np.array_equal(processed_VTA,label_VTA):
        if verbose: print('Test label 1 passed')
        return True
    if verbose: print('Test label 1 failed')
    return False

def test_label2(verbose : bool):
    if verbose : print('Test label 2', end='\t')
    VTA = np.array( [   [   [0, 0, 0],
                            [0, 1, 1], # region 1
                            [0, 0, 0]],
                        [   [0, 0 ,0],
                            [0, 0, 0],
                            [0, 0, 0]],
                        [   [0, 0, 0],
                            [0, 1, 1], # region 2
                            [0, 0, 0]]])
    label_VTA = np.array(   [   [   [0, 0, 0],
                                    [0, 1, 1],
                                    [0, 0, 0]],
                                [   [0, 0 ,0],
                                    [0, 0, 0],
                                    [0, 0, 0]],
                                [   [0, 0, 0],
                                    [0, 2, 2],
                                    [0, 0, 0]]])
    processed_VTA, num_regions = skimage.label(VTA, connectivity=3, return_num=True)
    #print(cleaned_VTAs)
    if np.array_equal(processed_VTA,label_VTA):
        if verbose: print('Test label 2 passed')
        return True
    if verbose: print('Test label 2 failed')
    return False

def test_label3(verbose : bool):
    if verbose : print('Test label 3', end='\t')
    VTA = np.array( [   [   [0, 0, 0],
                            [0, 1, 1], # region 1
                            [0, 1, 1]],
                        [   [0, 0 ,0],
                            [0, 0, 0],
                            [0, 0, 0]],
                        [   [0, 0, 0],
                            [0, 1, 1], # region 2
                            [0, 0, 0]]])
    label_VTA = np.array(   [   [   [0, 0, 0],
                                    [0, 1, 1],
                                    [0, 1, 1]],
                                [   [0, 0 ,0],
                                    [0, 0, 0],
                                    [0, 0, 0]],
                                [   [0, 0, 0],
                                    [0, 2, 2],
                                    [0, 0, 0]]])
    processed_VTA, num_regions = skimage.label(VTA, connectivity=3, return_num=True)
    #print(cleaned_VTAs)
    if np.array_equal(processed_VTA,label_VTA):
        if verbose : print('Test label 3 passed')
        return True
    if verbose : print('Test label 3 failed')
    return False

# ----------------------------------------------------------------
if __name__ == "__main__":
    verbose = True
    run_clean_tests(verbose)
    run_region_tests(verbose)
    run_label_tests(verbose)

# ----------------------------------------------------------------

def check_cleaned(VTAs):
    for i, VTA in enumerate(VTAs):
        # labelling again the VTA to double-check if cleaning worked
        label_img, num_regions = skimage.label(VTA, connectivity=3, return_num=True)
        regions = skimage.regionprops(label_img)
        if len(regions) > 0:
            # get biggest area
            area_check = regions[0].area
            #print(f'VTA #{i} : area {area_check}')
            #assert area_check >500, f'VTA #{i} of area {area_check} seems small'
        
        # if there is still more than one region, raise AssertionError
        if num_regions > 1:
            warnings.warn(f'Artefacts not removed in VTA #{i} : {num_regions} regions left instead of 1')

def log_deleted_artefacts(VTAs, clean_VTAs, resolution=None, log_folder=None, table=None, file_name='cleaning'):

    # ---- LOGGER SETUP --------------------------------------------------------------------------------------
    # datetime object containing current date and time
    now = datetime.now()
    
    # dd/mm/YY H:M:S
    dt_string = now.strftime("%Y_%m_%d_%H_%M_%S")
    #print("date and time =", dt_string)

    # Configure the logging format
    log_format = "%(levelname)s - %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=log_format)

    # Create a FileHandler to log messages to a file
    if log_folder is None:
        log_file = joinfile(LOG_DIR, f'cleaning/{file_name}_{dt_string}.log')
        csv_file = joinfile(LOG_DIR, f'cleaning/{file_name}_{dt_string}.csv')
    else:
        log_file = joinfile(log_folder, f'cleaning/{file_name}_{dt_string}.log')
        csv_file = joinfile(log_folder, f'cleaning/{file_name}_{dt_string}.csv')

    file_handler = logging.FileHandler(filename=log_file, mode='a')
    file_handler.setLevel(logging.DEBUG)

    # Create a Formatter to format the log messages
    formatter = logging.Formatter(log_format)
    file_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger = logging.getLogger()
    logger.addHandler(file_handler)

    logger.info('-- Deleted artefacts --')
    logger.info('Format : #VTA.#artefact   #voxels   volume[mm^3]')

    # ---- CSV SETUP --------------------------------------------------------------------------------------
    if table is not None:
        try:
            df = pd.read_csv(table)
        except:
            df = None
            logging.error(f'Cannot read table : {table}')
    df_dict = {
        'idx' : list(),
        'n_voxels' : list(),
        'volume mm^3' : list(),
        'source_nifti' : list()
    }
    # ---- VTA ITERATION --------------------------------------------------------------------------------------
    for i, (clean_img, img) in enumerate(zip(clean_VTAs, VTAs)):
        
        # diff is the artefacts erased
        diff = np.subtract(img,clean_img)
        if diff.any():

            try:
                source_nifti = df.at[i, 'massive_filename']
            except:
                logging.error(f'Could not retreive VTA {i} massive filename in table : {table}')
                source_nifti = ''

            label_img = skimage.label(diff, connectivity=3)
            regions = skimage.regionprops(label_img)
            for j, region in enumerate(regions):
                
                df_dict['idx'].append(f'{i}.{j}')
                df_dict['n_voxels'].append(f'{int(region.area):4d}')
                df_dict['volume mm^3'].append(f'{region.area*np.power((int(resolution)/1000),3):.2f}')
                df_dict['source_nifti'].append(source_nifti)
                
                if resolution is not None:
                    # print which VTA is it, and the size of the artefact
                    logging.info(f'Artefact {i}.{j+1} : {int(region.area):4d} voxels ({region.area*np.power((int(resolution)/1000),3):.2f} mm^3)')
                else:
                    logging.info(f'Artefact {i}.{j+1} : {int(region.area):4d} voxels')

    # ---- SAVING CSV --------------------------------------------------------------------------------------

    df_log = pd.DataFrame(df_dict)
    df_log.to_csv(csv_file, index=False)


# ----------------------------------------------------------------

def add_sphere_to_array(arr, position, radius):
    pass