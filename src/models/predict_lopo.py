import numpy as np
import os
import argparse
from importlib import import_module
import importlib.util
import sys
#sys.path.append("/media/brainstimmaps/DATA/2009_DeepMaps01/04_Source/01_Development/deepmaps/")
from datetime import datetime
from pathlib import Path
import warnings
import copy
warnings.filterwarnings('ignore') 
from tqdm import tqdm

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
from sklearn.model_selection import StratifiedShuffleSplit

from src.misc.joinfile import joinfile
from src.models.constants import models_meta
from src.metrics.metrics import Metrics, ModelPerformancePredict
from dir_const import DATA_PROCESSED_DIR, MODELS_DIR, SRC_DATASET_DIR
from src.models.train_model import get_model_specific_kwargs, get_sample_weights, WeightClipper, reset_weights

def get_three_loaders(dataset, batch_size, train_sampler, valid_sampler, test_sampler):
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=train_sampler)
    valid_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=valid_sampler)
    test_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=test_sampler)
    return train_loader, valid_loader, test_loader

def get_two_loaders(dataset, batch_size, train_sampler, test_sampler):
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=train_sampler)
    test_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=test_sampler)
    return train_loader, test_loader

def leave_one_patient_out(table, patient_id):
    train_ids = table[table['patient'] != patient_id].index.tolist()
    test_ids = table[(table['patient'] == patient_id) & (table['noisy'] == False)].index.tolist()
    #print(train_ids)
    #print(test_ids)
    return train_ids, test_ids

def leave_one_patient_out_center(table, patient_id, center):
    if center == 'merged':
        train_ids = table[table['patient'] != patient_id].index.tolist()
        test_ids = table[table['patient'] == patient_id].index.tolist()
    else:
        table_c = table[table['center'] == center]
        train_ids = table_c[table_c['patient'] != patient_id].index.tolist()
        test_ids = table_c[table_c['patient'] == patient_id].index.tolist()
    #print(train_ids)
    #print(test_ids)
    return train_ids, test_ids

def leave_one_center_out(table, train_center):
    train_ids = table[table['center'] == train_center].index.tolist()
    test_ids = table[table['center'] != train_center].index.tolist()
    #print(train_ids)
    #print(test_ids)
    return train_ids, test_ids

def split_train_valid(ids, labels, valid_size=0.2):
    sss = StratifiedShuffleSplit(n_splits=2, test_size=valid_size)
    train_ids, valid_ids = next(sss.split(ids, labels))
    train_val, train_count = np.unique(labels[train_ids], return_counts=True)
    valid_val, valid_count = np.unique(labels[valid_ids], return_counts=True)
    print(train_count, valid_count)
    return train_ids, valid_ids


def predict(args):
    # Device configuration  
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Choice of evaluation metric 
    EVAL_METRIC = Metrics.MCC

    verbose = False

    # ArgumentParser
    model_name = args.name
    assert model_name in models_meta.keys(), 'Model does not exist'
    version = args.version
    assert version in models_meta[model_name]['versions'].keys(), 'Version does not exist'

    # Unpacking model meta dict
    model_hyperparams = models_meta[model_name]['versions'][version]
    model_class_name = models_meta[model_name]["class_name"]
    criterion_name = models_meta[model_name]['criterion']
    dataset_module_name = models_meta[model_name]['dataset_module_name']
    dataset_class_name = models_meta[model_name]['dataset_class_name']
    model_print_name = models_meta[model_name]["print_name"]

    # Making the dataset arguments for different models
    #center = args.center
    #hemisphere = args.hemisphere
    #space = args.space
    #resolution = model_hyperparams['resolution']

    # Get model-specific keyword arguments
    model_specific_kwargs = get_model_specific_kwargs(dataset_module_name, args)

    # Loading the dataset class
    dataset_spec = importlib.util.spec_from_file_location(
        dataset_module_name,
        joinfile(SRC_DATASET_DIR, f'{dataset_module_name}.py'))
    dataset_module = importlib.util.module_from_spec(dataset_spec)
    sys.modules[dataset_module_name] = dataset_module
    dataset_spec.loader.exec_module(dataset_module)
    dataset = getattr(dataset_module, dataset_class_name)(
        input_folder=None, 
        space='stn_space_3sigma',
        augmentation_type=model_hyperparams['augmentation_type'], 
        center='merged',
        hemisphere='flipped',
        resolution=model_hyperparams['resolution'],
        interpolation=model_hyperparams['interpolation'],
        label_threshold=model_hyperparams['label_threshold'],
        noise_factor=model_hyperparams['noise_factor'],
        normalize_projections=False,
        tuning=False,
        tweening=True,
        n_split=1,
        **model_specific_kwargs)
    modelargs = dataset.get_model_args()

    # Get model library
    model_class = getattr(import_module(model_name), f'{model_class_name}')
    
    # Get hyper-parameters from model instance
    try:
        EPOCHS = model_hyperparams['epochs']
    except KeyError:
        EPOCHS = 10
    BATCH_SIZE = model_hyperparams['batch_size']
    LEARNING_RATE = model_hyperparams['learning_rate']
    OPTIMIZER = model_hyperparams['optimizer']

    # load table
    #input_table = joinfile(DATA_PROCESSED_DIR, f'{space}/{center}/{hemisphere}/VTAs/{resolution}um_table.csv')
    table = dataset.get_df()

    # get patientID list
    patients = table['patient'].unique()
    centers = table['center'].unique()

    # Getting datetime
    now = datetime.now()
    dt_string = now.strftime("%y.%m.%d.%H:%M:%S")

    # Create model performance metrics object
    perf = ModelPerformancePredict()
    perf.print_init(model_print_name, version)
    for n_center, center in enumerate(tqdm((*centers, 'merged'), 
                          colour = 'green', 
                          desc = 'Center', 
                          position = 0, 
                          leave = False)):
        for n_patient, patient in enumerate(tqdm(patients, 
                            colour = 'blue', 
                            desc = 'Patient', 
                            position = 1, 
                            leave = False)):

            # Split the train_valid ids form leave-on-out into train and valid ids
            train_valid_ids, test_ids = leave_one_patient_out_center(table, patient, center)
            labels = dataset.get_y()[train_valid_ids]
            train_ids, valid_ids = split_train_valid(train_valid_ids, labels)
            #print(f'Patient #{n_patient}, ID = {patient_id}')
            #print(f'Train ids : {len(train_ids)}')
            #print(f'Valid ids : {len(valid_ids)}')
            #print(f'Test ids : {len(test_ids)}')
            #print(f'Total : {len(train_ids) + len(valid_ids) + len(test_ids)}')
            
            # Creating PT data samplers and loaders:
            sample_weights = get_sample_weights(dataset, train_ids)
            train_sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_ids), replacement=True)
            valid_sampler = SubsetRandomSampler(valid_ids)
            test_sampler = SubsetRandomSampler(test_ids)

            # Normalizing independently each dataset split
            dataset.normalize(ids=train_ids)
            dataset.normalize(ids=valid_ids)
            dataset.normalize(ids=test_ids)

            # Loading the loaders
            train_loader , valid_loader, test_loader = get_three_loaders(dataset, BATCH_SIZE, train_sampler, valid_sampler, test_sampler)

            # Reloading the model at each patient
            model = model_class(**modelargs, **model_hyperparams, version=version).to(device)

            # Getting the weight decay hyperparameter for the optimizer
            try:
                weight_decay = model_hyperparams['weight_decay']
            except KeyError:
                weight_decay = .1

            # Defining loss, optimizer and scheduler
            criterion = getattr(nn, criterion_name)()
            optimizer = getattr(torch.optim, OPTIMIZER)(model.parameters(), lr=LEARNING_RATE, weight_decay=weight_decay)
            lambda1 = lambda epoch: 0.6 ** epoch
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

            # Early stopping
            min_loss = 100.0
            try:
                patience = model_hyperparams['patience']
            except:
                patience = 2

            triggertimes = 0 
            try:
                alpha_gl = model_hyperparams['alpha_gl']
            except KeyError:
                alpha_gl = 2.

            # Setting up the clipper
            clipper = WeightClipper()

            # Start training
            model.train()
            perf.start_train()

            for epoch in tqdm(range(EPOCHS), 
                                    colour = 'cyan', 
                                    desc = 'Epochs ', 
                                    position = 2, 
                                    leave = False):
                        
                # Set current loss value
                #current_loss = 0.0
                perf.start_epoch()
                # Iterating over batch
                #n_total_steps = len(train_loader)
                train_loss = 0.0
                for train_batch, (inputs, labels, indices) in enumerate(tqdm(train_loader, 
                                                        colour = 'white', 
                                                        desc = 'Batches', 
                                                        position = 3, 
                                                        leave = False)):
                    # --- TRAINING BATCH ITERATION --------------------------------
                    #_, label_counts = np.unique(labels, return_counts=True)
                    #labels_counts_total += label_counts
                    indices = [int(i) for i in indices]
                    
                    #iter = epoch * n_total_steps + train_batch

                    # Converting tensor to float
                    inputs = inputs.float().to(device)
                    labels = labels.float().to(device)

                    # Forward pass
                    outputs = model(inputs) # compute outputs
                    loss = criterion(outputs, labels) # evaluate loss
                    #a = list(model.parameters())[0].clone()
                    # Backward and optimize
                    optimizer.zero_grad() # flush previous gradients
                    loss.backward() # compute gradients
                    optimizer.step() # optimize parameters
                    #b = list(model.parameters())[0].clone()
                    #print(torch.equal(a.data, b.data))
                    model.apply(clipper)

                    # Writing loss
                    #n_samples = inputs.size(dim=0)
                    train_loss += loss.item()
                    #perf.add_training_loss(patient_id, epoch, train_batch, iter, loss.item())

                    # Writing accuracy
                    #outputs = torch.sigmoid(outputs) # output [-inf, inf] -> [0, 1]
                    #outputs = torch.round(outputs) # [0, 1] -> {0 or 1}


                    #outputs_dummy = [0] * len(inputs)
                    #labels_dummy = [0] * len(labels)
                    #perf.add_training_predictions(labels.detach().cpu().numpy(), outputs.detach().cpu().numpy(), indices)
                    #perf.add_training_predictions(labels_dummy, outputs_dummy, indices)
                # --- END OF TRAINING BATCH ITERATION -----------------------------
                scheduler.step()
                if verbose : print(f'Last LR : {scheduler.get_last_lr()}')
                if verbose : print(f'Train loss : {train_loss/len(train_loader)}')
                #print(labels_counts_total)
                #perf.stop_train()
                model.eval()
                perf.start_test()
                with torch.no_grad():
                    predictions = []
                    labels = []
                    current_loss, val_samples = 0.0, 0.0
                    
                    val_steps = len(valid_loader)
                    for i, (images, labels_, indices) in enumerate(valid_loader):
                        # --- VALIDATION BATCH ITERATION ----------------------
                        iter = epoch*val_steps + i
                        images = images.to(device).float()
                        labels_ = labels_.to(device).float()

                        outputs = model(images) # [-inf, inf]
                        loss = criterion(outputs, labels_) # sigmoid applied to outputs -> [0,1]
                        
                        #perf.add_testing_loss(fold, epoch, i, iter, loss.item())
                        outputs = torch.sigmoid(outputs)
                        #outputs_rounded = torch.round(outputs)

                        #average_epoch_testing_loss += loss.item()
                        predictions.extend(outputs.cpu().numpy())
                        labels.extend(labels_.cpu().numpy())
                        current_loss += (loss.item()*len(images))
                        val_samples += len(images)
                        # --- END OF VALIDATION BATCH ITERATION ---------------
                    current_loss /= val_samples
                    if verbose : print(f'Validation loss : {current_loss}')
                    #perf.add_testing_predictions(labels, predictions)

                    # Early Stopping
                    if current_loss*1.05 < min_loss:
                        min_loss = current_loss
                        best_model = copy.deepcopy(model)
                    generalization = 100*((current_loss/min_loss)-1)
                    if generalization > alpha_gl:
                        triggertimes += 1
                        if triggertimes >= patience:
                            if verbose : print('break')
                            break
            # --- TEST LOOP -------------------------------------------------------

            best_model.eval()
            with torch.no_grad():

                #perf.start_test()
                test_loss = 0.0
                val_steps = len(test_loader)
                for test_batch, (images, labels_, indices) in enumerate(test_loader):
                    # --- TEST BATCH ITERATION ------------------------------------
                    indices = [int(i) for i in indices]
                    
                    #iter = (epoch//VALIDATION_INTERVAL)*val_steps + i
                    images = images.to(device).float()
                    labels_ = labels_.to(device).float()

                    outputs = best_model(images) # [-inf, inf]
                    loss = criterion(outputs, labels_) # sigmoid applied to outputs -> [0,1]
                    test_loss += loss.item()
                    outputs = torch.sigmoid(outputs)
                    outputs_rounded = torch.round(outputs)
        
                    #outputs_dummy = [0] * len(images)
                    #labels_dummy = [0] * len(labels_)
                    if center == 'merged':
                        cst_kw = 'merged_'
                    else:
                        cst_kw = ''

                    for i in range(len(indices)):
                        table.loc[indices[i],f'{cst_kw}prediction_rounded'] = outputs_rounded[i].detach().cpu().numpy().astype(int)
                        table.loc[indices[i],f'{cst_kw}prediction'] = outputs[i].detach().cpu().numpy().astype(float)

                    #perf.add_testing_predictions(labels_.detach().cpu().numpy(), outputs.detach().cpu().numpy(), indices)
                    #perf.add_testing_predictions(labels_dummy, outputs_dummy, indices)
                    # --- END OF VALIDATION BATCH ITERATION ---------------
                if verbose :    print(f'Test loss : {test_loss/len(test_loader)}')
                #print('Average Epoch Testing Loss : {}'.format(average_epoch_testing_loss), end = '\r')
                #perf.stop_test()

                # --- END OF VALIDATION ---------------------------------------

                #perf.end_epoch(patient_id, epoch)  
                
                # --- END OF EPOCH ------------------------------------------------   
            
            #perf.print_current_fold()   

            # Fold timer
            #perf.end_patient(n_patient, epoch)
            
            #perf.timer.print_fold()

            # --- END OF FOLD -----------------------------------------------------
        
        #perf.compute_overall_metrics()
        #perf.print_train_results(folds=K_FOLD)

        # saving model performance metrics
        #path = joinfile(MODELS_DIR, f'predict/perf/{model_class_name}{version}/{dataset_type}/{center}/{hemisphere}/{resolution}um/{dt_string}.pkl')
        #Path(path).parents[0].mkdir(parents=True, exist_ok=True)
        #perf.save(path)

    path = joinfile(MODELS_DIR, f'predict_lopo/{model_class_name}{version}/datasets/{dt_string}.csv')
    Path(path).parents[0].mkdir(parents=True, exist_ok=True)
    dataset.export_df(path)

    path = joinfile(MODELS_DIR, f'predict_lopo/{model_class_name}{version}/perf/{dt_string}.csv')
    Path(path).parents[0].mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Launch the training of a model"
    )
    parser.add_argument("--name", type=str, default='proj_net')
    parser.add_argument("--version", type=str, default='Merged4',
                        help="Version of the model")

    parser.add_argument("--center", type=str, default ='merged',
                        help="Center of the mapping session")
    parser.add_argument("--space", type=str, default ='stn_space_3sigma',
                        help="Space")
    parser.add_argument("--hemisphere", type=str, default='flipped', 
                        help="Side of dataset : (both_hemisphere, right_hemisphere, left_hemisphere, flipped)")


    parser.add_argument("--input_table", type=str, default ='data/',
                        help="Center of the mapping session")
    
    args = parser.parse_args()
    
    predict(args)