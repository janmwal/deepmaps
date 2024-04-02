from cgi import test
import numpy as np
import warnings

def run_vta_tests(VTAs):
    print('=== Running tests about VTAs ===')
    print_vta_density(VTAs)
    #test_vta_binary(VTAs, warn = True)
    test_vta_empty(VTAs)
    print()

def print_vta_density(VTAs_):
    VTAs = VTAs_.copy()
    if not test_vta_binary(VTAs_):
        #print("VTAs not fully binary")
        VTAs[VTAs != 0.] = 1.
    print(f'Density : {(100*np.sum(VTAs)/np.product(VTAs.shape)):.6f}%')

def test_vta_binary(VTAs_, warn : bool = True):
    VTAs = VTAs_.copy()
    flag = False
    unique_dict = {}
    unique_vta = []
    for i, VTA in enumerate(VTAs):
        VTA[VTA == 1.] = 0.
        non_bin_uniques = np.unique(VTA.flatten())
        non_bin_uniques = [i for i in non_bin_uniques if i != 0.]
        if non_bin_uniques is not None:
            for unique in non_bin_uniques:
                sum_unique = int(np.sum(VTA[VTA == unique])//unique)
                flag = True
                unique_vta.append(i)
                if unique in unique_dict.keys():
                    unique_dict[unique] += sum_unique
                else:
                    unique_dict[unique] = sum_unique
                if warn:
                    warnings.warn(f'VTA #{i} contains : {sum_unique} times the number {unique}.')
    print(f'Non-binary values collected : {unique_dict}')
    print(f'VTAs that contain anything else than one or zero : {unique_vta}')
    return not flag

def test_vta_empty(VTAs_, warn : bool = True):
    VTAs = VTAs_.copy()
    flag = False
    empty_id_list = []
    for i, VTA in enumerate(VTAs):
        if (VTA == 0.).all():
            #print(f'{i} : {np.unique(VTA)}')
            flag = True
            empty_id_list.append(i)
    if flag:
        print(f'Empty VTAs : {empty_id_list}')
    print(f'Non-empty VTA ratio : {100*(1-(len(empty_id_list)/len(VTAs))):.2f}%')
    return flag

