# Standard libraries --------------------------------
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
# 3rd party libraries -------------------------------
import torch
import torch.nn as nn
from tqdm import tqdm
#import torchvision.transforms as transforms #UNUSED
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold

# Project libraries --------------------------------
from src.metrics.metrics import binary_acc, get_accuracy_tensor, get_balanced_accuracy_tensor, Metrics, ModelPerformanceTrain
from src.misc.joinfile import joinfile
from src.models.constants import *
from dir_const import MODELS_DIR, SRC_DATASET_DIR

DIR = os.path.dirname(os.path.realpath(__file__))

###############################################################################
# --- FORMULAS ---------------------------------------------------------------#
###############################################################################

def reverse_bce(loss):
    
    loss = np.where(loss > 1e-10, loss, 1e-13)
    result = - np.log(np.exp(loss)-1, where = loss > 1e-10)
    return result

###############################################################################
# --- MODEL WEIGHTS ----------------------------------------------------------#
###############################################################################

def reset_weights(m):
    '''
        Try resetting model weights to avoid
        weight leakage.
    '''
    # TODO : test if this works properly !!
    
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            #print(f'Reset trainable parameters of layer = {layer}')
            layer.reset_parameters()
    #print('Model weights reset')

class WeightClipper(object):

    def __init__(self, frequency=5):
        self.frequency = frequency

    def __call__(self, module):
        # filter the variables to get the ones you want
        if hasattr(module, 'weight'):
            w = module.weight.data
            w = w.clamp(-1,1)
            module.weight.data = w

###############################################################################
# --- UNBALANCED DATASET -----------------------------------------------------#
###############################################################################

def get_sample_weights(dataset, indices):
    subset_labels = dataset._get_labels()
    #print((labels == 0.).sum())
    #print((labels == 1.).sum())
    class_weights = [1./(subset_labels == 0.).sum(), 1./(subset_labels == 1.).sum()]
    #print(class_weights)
    sample_weights = [0] * len(dataset)
    for img, label, idx in dataset:
        if idx in indices:
            class_weight = class_weights[int(label)]
            sample_weights[idx] = class_weight
        else:
            sample_weights[idx] = 0.
    #print([f'{s}, {l} \n' for (s, l) in zip(sample_weights, labels)])
    return sample_weights

def get_class_weights(dataset, indices):
    labels = dataset._get_labels()[indices]
    class_weights = [1./(labels == 0.).sum(), 1./(labels == 1.).sum()]

    return class_weights

###############################################################################
# --- LOADERS ----------------------------------------------------------------#
###############################################################################

def get_loader(dataset, batch_size, train_sampler, test_sampler):
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=train_sampler)
    test_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=test_sampler)
    return train_loader, test_loader

def get_model_specific_kwargs(dataset_module_name, args):   
        if dataset_module_name == 'vta_dataset':
            return {'dummy' : None}
        elif dataset_module_name == 'props_dataset': 
            #return {'props_type' : args.props_type}
            return {'dummy' : None}
        elif dataset_module_name == 'proj_dataset':
            return {'dummy' : None}
        else:
            raise ValueError('Given dataset_module_name does not exist')

###############################################################################
# --- TRAINER ----------------------------------------------------------------#
###############################################################################

def train(args):    
    # Device configuration  
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Hyper-parameters 
    EPOCHS = 10
    SHUFFLE_DATASET = True
    K_FOLD = 5
    EVAL_METRIC = Metrics.BAL_ACCURACY
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
    augmentation_type = model_hyperparams['augmentation_type']
    center = args.center
    hemisphere = args.hemisphere
    resolution = args.resolution
    interpolation = model_hyperparams['interpolation']

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
    model = model_class(version=version, **modelargs, **model_hyperparams,).to(device)

    # Get hyper-parameters from model instance
    BATCH_SIZE = model_hyperparams['batch_size']
    LEARNING_RATE = model_hyperparams['learning_rate']
    OPTIMIZER = model_hyperparams['optimizer']

    # Defining loss and optimizer
    criterion = getattr(nn, criterion_name)()
    optimizer = getattr(torch.optim, OPTIMIZER)(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-1)

    # Define the K-fold Cross Validator
    kfold = StratifiedKFold(n_splits=K_FOLD, shuffle=SHUFFLE_DATASET)

    # Printing loss per mini-batch
    verbose = False
    n_step_show = 20

    # Setting up the clipper
    clipper = WeightClipper()

    # Saving options
    save_each_fold = False
    save_best_model = False
    
    # Getting datetime 
    now = datetime.now()
    dt_string = now.strftime("%y.%m.%d.%H:%M:%S")

    # Create model performance metrics object
    perf = ModelPerformanceTrain(multi_threshold=True, n_thresholds=11)
    perf.print_init(model_print_name, EVAL_METRIC, version)

    # K-fold Cross Validation model evaluation
    for fold, (train_ids, test_ids) in enumerate(tqdm(kfold.split(dataset, dataset.get_y()), 
                                                      colour = 'blue', 
                                                      desc = 'Folds  ', 
                                                      position = 0,
                                                      leave = False,
                                                      total = kfold.get_n_splits())):
        # ----- START OF ONE FOLD ---------------------------------------------

        perf.new_fold(fold)
        #dataset.test_stratified_kfold(train_ids, test_ids)

        sample_weights = get_sample_weights(dataset, train_ids)

        # Creating PT data samplers and loaders:
        train_sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_ids), replacement=True)
        test_sampler = SubsetRandomSampler(test_ids)

        dataset.normalize(ids=train_ids)
        dataset.normalize(ids=test_ids)
        
        train_loader , test_loader = get_loader(dataset, BATCH_SIZE, train_sampler, test_sampler)

        # Reset model parameters
        model.apply(reset_weights)
        model.train()
   
        n_total_steps = len(train_loader)

         # --- TRAINING LOOP --------------------------------------------------
        perf.start_train()
        for epoch in tqdm(range(EPOCHS), 
                          colour = 'cyan', 
                          desc = 'Epochs ', 
                          position = 1, 
                          leave = False):
            
            # Set current loss value
            current_loss = 0.0
            perf.start_epoch()
            # Iterating over batch
            n_total_steps = len(train_loader)
            for i, (inputs, labels, indices) in enumerate(tqdm(train_loader, 
                                                      colour = 'white', 
                                                      desc = 'Batches', 
                                                      position = 2, 
                                                      leave = False)):
                # --- TRAINING BATCH ITERATION --------------------------------
                #_, label_counts = np.unique(labels, return_counts=True)
                #labels_counts_total += label_counts

                iter = epoch * n_total_steps + i

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
                n_samples = inputs.size(dim=0)
                current_loss += loss.item()
                perf.add_training_loss(fold, epoch, i, iter, loss.item())

                # Writing accuracy
                outputs = torch.sigmoid(outputs) # output [-inf, inf] -> [0, 1]
                outputs_rounded = torch.round(outputs) # [0, 1] -> {0 or 1}

                perf.add_training_predictions(labels.detach().cpu().numpy(), outputs.detach().cpu().numpy())

                # Priting loss
                if (i+1) % n_step_show == n_step_show-1 and verbose:
                    # TODO : function for this print here
                    acc = binary_acc(outputs, labels)
                    print (f'\tEpoch [{epoch+1}/{EPOCHS}], Step [{i+1}/'
                    f'{n_total_steps}], Loss: '
                    f'{current_loss/float(n_step_show):.4f}, Accuracy : {acc} %')
                    current_loss = 0.0

            # --- END OF TRAINING BATCH ITERATION -----------------------------
            #print(labels_counts_total)
            perf.stop_train()

            # --- VALIDATION LOOP ---------------------------------------------
            #average_epoch_testing_loss = 0
            if (epoch + 1) % VALIDATION_INTERVAL == 0:

                model.eval()
                with torch.no_grad():
                    predictions = []
                    labels = []

                    perf.start_test()
                    val_steps = len(test_loader)
                    for i, (images, labels_, indices) in enumerate(test_loader):
                        # --- VALIDATION BATCH ITERATION ----------------------
                        iter = (epoch//VALIDATION_INTERVAL)*val_steps + i
                        images = images.to(device).float()
                        labels_ = labels_.to(device).float()

                        outputs = model(images) # [-inf, inf]
                        loss = criterion(outputs, labels_) # sigmoid applied to outputs -> [0,1]
                        
                        perf.add_testing_loss(fold, epoch, i, iter, loss.item())
                        outputs = torch.sigmoid(outputs)
                        #outputs_rounded = torch.round(outputs)

                        #average_epoch_testing_loss += loss.item()
                        predictions.extend(outputs.cpu().numpy())
                        labels.extend(labels_.cpu().numpy())
                        # --- END OF VALIDATION BATCH ITERATION ---------------

                    #average_epoch_testing_loss /= len(test_loader)
                    #print('Average Epoch Testing Loss : {}'.format(average_epoch_testing_loss), end = '\r')
                    perf.stop_test()

                # --- END OF VALIDATION ---------------------------------------
                perf.add_testing_predictions(labels, predictions)

            perf.end_epoch(fold, epoch)  
            
            # --- END OF EPOCH ------------------------------------------------   
        
        #perf.print_current_fold()   

        # Fold timer
        perf.end_fold()

        if save_each_fold:
            perf.timer.start()
            path = joinfile(MODELS_DIR, f'{model_class_name}{version}/{center}/{augmentation_type}/folds/{fold+1}.pth')
            Path(path).parents[0].mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), path)
            perf.timer.stop()
            # print(f'Model saved in {(perf.timer.get_time()):.2f}s')

        # # Saving the model
        # if perf.is_best_model(fold) and save_best_model:
        #     perf.timer.start()
        #     path = joinfile(MODELS_DIR, f'{model_class_name}{version}/{center}/bestmodel.pth')
        #     Path(path).parents[0].mkdir(parents=True, exist_ok=True)
        #     torch.save(model.state_dict(), path)
        #     perf.timer.stop()
        #     # print(f'Best model based on {EVAL_METRIC.value} saved in '
        #     #       f'{(perf.timer.get_time()):.2f}s')

        
        #perf.timer.print_fold()

        # --- END OF FOLD -----------------------------------------------------
    
    perf.compute_overall_metrics()
    perf.print_train_results(folds=K_FOLD)

    # saving model performance metrics
    path = joinfile(MODELS_DIR, f'train/{model_class_name}{version}/{center}/{augmentation_type}/{hemisphere}/{interpolation}/{resolution}um/perf/{dt_string}.pkl')
    Path(path).parents[0].mkdir(parents=True, exist_ok=True)
    perf.save(path)

###############################################################################
# --- MAIN -------------------------------------------------------------------#
###############################################################################
   
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Launch the training of a model"
    )
    parser.add_argument("--name", type=str, default='props_net')
    parser.add_argument("--version", type=str,
                        help="Version of the model")

    parser.add_argument("--center", type=str, default ='bern',
                        help="Center of the mapping session")
    parser.add_argument("--augmentation_type", type=str, default='augmented_full',
                        help="Type of dataset : augmented_part, augmented_full, mapping")
    parser.add_argument("--hemisphere", type=str, default='flipped', 
                        help="Side of dataset : (both_hemisphere, right_hemisphere, left_hemisphere, flipped)")
    parser.add_argument("--resolution", type=int, default=250,
                        help="Resolution to train on (um).")
    parser.add_argument("--interpolation", type=str, default ='linear_interp',
                        help="Type of interpolation : linear_interp, step_interp, no_interp")
    
    parser.add_argument("--threshold_labels", type=float, 
                        help="Threshold value of the binarization")
    
    parser.add_argument("--props_type", type=str, default ='simple_props',
                        help="Type of proprieties")
    
    args = parser.parse_args()
    train(args)
