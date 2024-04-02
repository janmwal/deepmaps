import numpy as np
import os
import argparse
from importlib import import_module
import importlib.util
import sys
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore') 
from tqdm import tqdm

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler

from misc.joinfile import joinfile
from constants import models_meta
from metrics.metrics import Metrics, ModelPerformancePredict
from dir_const import DATA_PROCESSED_DIR, MODELS_DIR, SRC_DATASET_DIR
from models.train_model import get_model_specific_kwargs, get_sample_weights, WeightClipper, reset_weights

def get_loaders(dataset, batch_size, train_sampler, test_sampler):
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=train_sampler)
    test_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=test_sampler)
    return train_loader, test_loader

def leave_one_patient_out(table, patient_id):
    train_ids = table[table['patientID'] != patient_id].index.tolist()
    test_ids = table[table['patientID'] == patient_id].index.tolist()
    #print(train_ids)
    #print(test_ids)
    return train_ids, test_ids

def predict(args):
    # Device configuration  
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Hyper-parameters 
    EVAL_METRIC = Metrics.MCC
    VALIDATION_INTERVAL = 1

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
    center = args.center
    hemisphere = args.hemisphere
    resolution = model_hyperparams['resolution']
    interpolation_type = model_hyperparams['interpolation']
    augmentation_type=model_hyperparams['augmentation_type']


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
        augmentation_type=model_hyperparams['augmentation_type'], 
        center=center,
        hemisphere=hemisphere,
        resolution=model_hyperparams['resolution'],
        interpolation=model_hyperparams['interpolation'],
        label_threshold=model_hyperparams['label_threshold'],
        **model_specific_kwargs)
    modelargs = dataset.get_model_args()

    # Model creation
    model_class = getattr(import_module(model_name), f'{model_class_name}')
    model = model_class(**modelargs, **model_hyperparams, version=version).to(device)

    # Get hyper-parameters from model instance
    try:
        EPOCHS = model_hyperparams['epochs']
    except KeyError:
        EPOCHS = 10

    BATCH_SIZE = model_hyperparams['batch_size']
    LEARNING_RATE = model_hyperparams['learning_rate']
    OPTIMIZER = model_hyperparams['optimizer']

    # Defining loss and optimizer
    criterion = getattr(nn, criterion_name)()
    optimizer = getattr(torch.optim, OPTIMIZER)(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-1)

    # Setting up the clipper
    clipper = WeightClipper()

    # load table
    input_table = joinfile(DATA_PROCESSED_DIR, f'{center}/{augmentation_type}/{hemisphere}/processed_tables/{interpolation_type}/processed_table.csv')
    table = pd.read_csv(input_table)
    # get patientID list
    patients = table['patientID'].unique()

    # Getting datetime
    now = datetime.now()
    dt_string = now.strftime("%y.%m.%d.%H:%M:%S")

    # Create model performance metrics object
    perf = ModelPerformancePredict()
    perf.print_init(model_print_name, EVAL_METRIC, version)

    for n_patient, patient_id in enumerate(tqdm(patients, 
                          colour = 'blue', 
                          desc = 'Patient', 
                          position = 0, 
                          leave = False)):
        train_ids, test_ids = leave_one_patient_out(table, patient_id)
        #print(f'Patient #{n_patient}, ID = {patient_id}')
        #print(f'Train ids : {train_ids}')
        #print(f'Test ids : {test_ids}')

        # Creating PT data samplers and loaders:
        sample_weights = get_sample_weights(dataset, train_ids)
        train_sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_ids), replacement=True)
        test_sampler = SubsetRandomSampler(test_ids)
        dataset.normalize(ids=train_ids)
        dataset.normalize(ids=test_ids)
        train_loader , test_loader = get_loaders(dataset, BATCH_SIZE, train_sampler, test_sampler)

        model.apply(reset_weights)
        model.train()
        perf.start_train()

        for epoch in tqdm(range(EPOCHS), 
                          colour = 'cyan', 
                          desc = 'Epochs ', 
                          position = 1, 
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
                                                      position = 2, 
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
                
                # Backward and optimize
                optimizer.zero_grad() # flush previous gradients
                loss.backward() # compute gradients
                optimizer.step() # optimize parameters

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
            #print(labels_counts_total)
            #perf.stop_train()
            print(f'Train loss : {train_loss/len(train_loader)}')
            # --- VALIDATION LOOP ---------------------------------------------

            model.eval()
            with torch.no_grad():
                test_loss = 0.0
                #perf.start_test()
                val_steps = len(test_loader)
                for i, (images, labels_, indices) in enumerate(test_loader):
                    # --- VALIDATION BATCH ITERATION ----------------------
                    indices = [int(i) for i in indices]
                    
                    #iter = (epoch//VALIDATION_INTERVAL)*val_steps + i
                    images = images.to(device).float()
                    labels_ = labels_.to(device).float()

                    outputs = model(images) # [-inf, inf]
                    loss = criterion(outputs, labels_) # sigmoid applied to outputs -> [0,1]
                    test_loss += loss.item()
                    outputs = torch.sigmoid(outputs)
                    outputs = torch.round(outputs)
        
                    #outputs_dummy = [0] * len(images)
                    #labels_dummy = [0] * len(labels_)
                    for i in range(len(indices)):
                        table.loc[indices[i],'prediction'] = outputs[i].detach().cpu().numpy().astype(int)
                    #perf.add_testing_predictions(labels_.detach().cpu().numpy(), outputs.detach().cpu().numpy(), indices)
                    #perf.add_testing_predictions(labels_dummy, outputs_dummy, indices)
                    # --- END OF VALIDATION BATCH ITERATION ---------------
            print(f'Test loss : {test_loss/len(test_loader)}')
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

    path = joinfile(MODELS_DIR, f'predict/{model_class_name}{version}/'
            f'{center}/{augmentation_type}/{hemisphere}/{interpolation_type}/'
            f'{resolution}um/csv/{dt_string}.csv')
    Path(path).parents[0].mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Launch the training of a model"
    )
    parser.add_argument("--name", type=str, default='props_net')
    parser.add_argument("--version", type=str,
                        help="Version of the model")

    parser.add_argument("--dataset_type", type=str,
                        help="Type of dataset : augmented_part, augmented_full, mapping")
    parser.add_argument("--center", type=str, default ='merged',
                        help="Center of the mapping session")
    parser.add_argument("--hemisphere", type=str, default='flipped', 
                        help="Side of dataset : (both_hemisphere, right_hemisphere, left_hemisphere, flipped)")
    parser.add_argument("--resolution", type=int, default=250,
                        help="Resolution to train on (um).")
    
    parser.add_argument("--threshold_labels", type=float, 
                        help="Threshold value of the binarization")
    
    parser.add_argument("--props_type", type=str, default ='simple_props',
                        help="Type of proprieties")

    parser.add_argument("--input_table", type=str, default ='data/',
                        help="Center of the mapping session")
    
    args = parser.parse_args()
    
    predict(args)