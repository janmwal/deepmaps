"""
Optuna example that optimizes multi-layer perceptrons using PyTorch.
In this example, we optimize the validation accuracy of fashion product recognition using
PyTorch and FashionMNIST. We optimize the neural network architecture as well as the optimizer
configuration. As it is too time consuming to use the whole FashionMNIST dataset,
we here use a small subset of it.
"""

import os
import importlib.util
import sys
from datetime import datetime
import pickle
from pathlib import Path
import copy

import optuna
from optuna.pruners import ThresholdPruner, BasePruner
from optuna.trial import TrialState
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

import numpy as np
from tqdm import tqdm
# Project libraries --------------------------------
from metrics.metrics import binary_acc, get_accuracy_tensor, Metrics, ModelPerformance, get_balanced_accuracy
from misc.joinfile import joinfile
from constants import *
from dir_const import MODELS_DIR, SRC_DATA_DIR, SRC_DATASET_DIR

EPOCHS = 10
MAX_BATCH_SIZE = 16
EVAL_METRIC = 'MCC'
OPTIMIZE_LOSS = False

def split_train_valid(ids, labels, valid_size=0.2):
    sss = StratifiedShuffleSplit(n_splits=2, test_size=valid_size)
    train_ids, valid_ids = next(sss.split(ids, labels))
    #train_val, train_count = np.unique(labels[train_ids], return_counts=True)
    #valid_val, valid_count = np.unique(labels[valid_ids], return_counts=True)
    #print(train_count, valid_count)
    return train_ids, valid_ids

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

def get_sample_weights(dataset, indices):
    subset_labels = dataset._get_labels()[indices]
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

class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()
    
    def forward(self, x):
        # Do your print / debug stuff here
        print(x.shape)
        return x

dispatcher = { 
    'Balanced Accuracy' : balanced_accuracy_score, 
    'F1 Score' : f1_score,
    'Precision' : precision_score,
    'Recall' : recall_score,
    'MCC' : matthews_corrcoef}


def compute_metric(labels, preds, func):
    try:
        return dispatcher[func](labels, preds)
    except:
        return "Invalid function"

def define_model(trial, img_shape):
    # We optimize the number of layers, hidden units and dropout ratio in each layer.
    verbose = False
    n_conv_layers = trial.suggest_int("n_conv_layers", 1, 4)
    layers = []
    conv_shape = img_shape['in_shape'][1:]
    in_channels = 1
    for i in range(n_conv_layers):
        if verbose: print(f'Conv layer {i}')   
        out_channels = trial.suggest_int(
            "n_units_conv_l{}".format(i), 
            2, 16)
        if verbose: print(f'\tChannels {out_channels}')
        layers.append(nn.Conv3d(
            in_channels=in_channels, out_channels=out_channels, 
            kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm3d(out_channels))
        if verbose: print('\tOut conv shape : {}'.format(conv_shape))
        in_channels = out_channels
    layers.append(nn.Flatten())

    #print('conv shape : {}'.format(conv_shape))
    #print('out channels : {}'.format(out_channels))
    product = np.prod(conv_shape)
    #print('conv shape product : {}'.format(product))
    in_features = (product*out_channels).astype(np.uint32)
    if verbose: 
        print('\n=======')
        print('Flatten features  : {}'.format(in_features))
        print('=======\n')
    n_fc_layers = trial.suggest_int("n_fc_layers", 1, 5)
    for i in range(n_fc_layers):
        if verbose: print(f'FC layer {i}') 
        out_features = trial.suggest_int(
            "n_units_fc_l{}".format(i), 
            4, 50)
        if verbose: print(f'\tChannels {out_features}')
        layers.append(nn.Linear(in_features, out_features))
        if i < n_fc_layers-1:
            # if we're not in the last layer
            layers.append(nn.ReLU())
        
        in_features = out_features
    if verbose: print()
    layers.append(nn.Linear(in_features, 1))
    return nn.Sequential(*layers)

def vtanet_opti2(trial, img_shape):
    # We optimize the number of layers, hidden units and dropout ratio in each layer.
    verbose = False
    n_conv_layers = trial.suggest_int("n_conv_layers", 1, 4)
    layers = []
    conv_shape = img_shape['in_shape'][1:]
    in_channels = 1
    for i in range(n_conv_layers):
        if verbose: print(f'Conv layer {i}')   
        out_channels = trial.suggest_int(
            "n_units_conv_l{}".format(i), 
            2, 16)
        if verbose: print(f'\tChannels {out_channels}')
        layers.append(nn.Conv3d(
            in_channels=in_channels, out_channels=out_channels, 
            kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm3d(out_channels))
        if verbose: print('\tOut conv shape : {}'.format(conv_shape))
        in_channels = out_channels
    layers.append(nn.Flatten())

    #print('conv shape : {}'.format(conv_shape))
    #print('out channels : {}'.format(out_channels))
    product = np.prod(conv_shape)
    #print('conv shape product : {}'.format(product))
    in_features = (product*out_channels).astype(np.uint32)
    if verbose: 
        print('\n=======')
        print('Flatten features  : {}'.format(in_features))
        print('=======\n')
    n_fc_layers = trial.suggest_int("n_fc_layers", 2, 6)
    for i in range(n_fc_layers):
        if verbose: print(f'FC layer {i}') 
        out_features = trial.suggest_int(
            "n_units_fc_l{}".format(i), 
            4, 50)
        if verbose: print(f'\tChannels {out_features}')
        layers.append(nn.Linear(in_features, out_features))
        if i < n_fc_layers-1:
            # if we're not in the last layer
            layers.append(nn.ReLU())
        dropout = trial.suggest_float(
            "dropout_fc_{}".format(i), 
            0.1, 0.8)
        layers.append(nn.Dropout1d(dropout))
        in_features = out_features
    if verbose: print()
    layers.append(nn.Linear(in_features, 1))
    return nn.Sequential(*layers)

def objective(trial):
    # Device configuration  
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #print(torch.cuda.current_device())
    
    # Loading the dataset
    model_name = 'vta_net'
    dataset_module_name = models_meta[model_name]['dataset_module_name']
    dataset_class_name = models_meta[model_name]['dataset_class_name']
    resolution = 500
    dataset_spec = importlib.util.spec_from_file_location(
        dataset_module_name,
        joinfile(SRC_DATASET_DIR, f'{dataset_module_name}.py'))
    dataset_module = importlib.util.module_from_spec(dataset_spec)
    sys.modules[dataset_module_name] = dataset_module
    dataset_spec.loader.exec_module(dataset_module)
    label_threshold = trial.suggest_float("label_treshold", 0.5, 0.8)
    augmentation_type = 'augmented_full'
    interpolation = 'linear_interp'
    
    dataset = getattr(dataset_module, dataset_class_name)(
        input_folder=None, 
        augmentation_type=augmentation_type, 
        center='bern',
        hemisphere='flipped',
        resolution=resolution,
        interpolation=interpolation,
        label_threshold=label_threshold,
        normalize_projections=False)
    img_shape = dataset.get_model_args()

    # Generate the model.
    #model = define_model(trial, img_shape).to(device)    
    criterion_name = models_meta[model_name]['criterion']

    # Generate the optimizers.
    #optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD", "Adamax"])
    optimizer_name = 'Adam'
    lr = trial.suggest_float("lr", 1e-7, 1e-4, log=True)
    

    chosen_batch_size = trial.suggest_categorical("batch_size", [4, 8, 16, 32])
    print('\n=======')
    print(f'Batch size : {chosen_batch_size}, Optimizer : {optimizer_name}, Learning Rate : {lr:.2e}')
    print('=======')
    if chosen_batch_size > MAX_BATCH_SIZE:
        batch_size = MAX_BATCH_SIZE
        accum_iter = chosen_batch_size // MAX_BATCH_SIZE
    else:
        batch_size = chosen_batch_size
        accum_iter = 1

    # Setting up the clipper
    clipper = WeightClipper()

    # Define the K-fold Cross Validator
    kfold = StratifiedKFold(n_splits=5, shuffle=True)
    metric_fold , loss_fold = [], []

    weightdecay = trial.suggest_float("weight_decay", 1e-2, 2, log=True)
    lambda_coeff = trial.suggest_float("lambda_coeff", 0.4, 1)
    alpha_early_stopping = trial.suggest_float("alpha_gl", 1., 20.)
    for fold, (train_valid_ids, test_ids) in enumerate(tqdm(kfold.split(dataset, dataset.get_y()), 
                                                      colour = 'blue', 
                                                      desc = 'Folds  ', 
                                                      position = 0,
                                                      leave = False,
                                                      total = kfold.get_n_splits())):
        labels = dataset.get_y()[train_valid_ids]
        train_ids, valid_ids = split_train_valid(train_valid_ids, labels)
        sample_weights = get_sample_weights(dataset, train_ids)

        # Creating PT data samplers and loaders:
        train_sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_ids), replacement=True)
        valid_sampler = SubsetRandomSampler(valid_ids)
        test_sampler = SubsetRandomSampler(test_ids)

        dataset.normalize(ids=train_ids)
        dataset.normalize(ids=valid_ids)
        dataset.normalize(ids=test_ids)

        # Get the dataset
        train_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, sampler=train_sampler)
        valid_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, sampler=valid_sampler)
        test_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, sampler=test_sampler)

        # Generate the model.
        model = vtanet_opti2(trial, img_shape).to(device)    
        criterion_name = models_meta[model_name]['criterion']
        
        optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr, weight_decay=weightdecay)
        criterion = getattr(nn, criterion_name)()
        
        lambda1 = lambda epoch: lambda_coeff ** epoch
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

        metric = []
        loss_ = []
        

        # Early stopping
        min_loss = 100.0
        patience = 2
        triggertimes = 0 
        

        # Training of the model.
        for epoch in tqdm(range(EPOCHS), colour = 'cyan', desc = 'Epochs ', position = 1, leave = False):
            model.train()
            for batch_idx, (inputs, labels, index) in enumerate(tqdm(train_loader, 
                                                            colour = 'white', 
                                                            desc = 'Batches', 
                                                            position = 2, 
                                                            leave = False)):
                # Converting tensor to float
                inputs = inputs.to(device).float()
                labels = labels.to(device).float()
                #print(f'Input shape : {inputs.shape}')
                #print(f'Label shape : {labels.shape}')
                # Forward pass
                #print(inputs.size())
                outputs = model(inputs) # compute outputs
                loss = criterion(outputs, labels) # evaluate loss

                #print(torch.cuda.memory_summary(device=device))

                loss.backward() # compute gradients

                if ((batch_idx + 1) % accum_iter == 0) or (batch_idx + 1 == len(train_loader)):
                    # Backward and optimize
                    optimizer.step() # optimize parameters
                    optimizer.zero_grad() # flush previous gradients        
                    model.apply(clipper)
            # Validation of the model.
            scheduler.step()
            model.eval()
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
                #print(f'Validation loss : {current_loss}')
                #perf.add_testing_predictions(labels, predictions)

                # Early Stopping
                if current_loss*1.05 < min_loss:
                    min_loss = current_loss
                    best_model = copy.deepcopy(model)
                generalization = 100*((current_loss/min_loss)-1)
                if generalization > alpha_early_stopping:
                    triggertimes += 1
                    if triggertimes >= patience:
                        #print('break')
                        break

        if OPTIMIZE_LOSS:
            losses = []
        else:
            predictions, labels = [], []
        best_model.eval()
        with torch.no_grad():
            for batch_idx, (images, labels_, index) in enumerate(test_loader):
                images = images.to(device).float()
                labels_ = labels_.to(device).float()

                outputs = best_model(images)
                loss = criterion(outputs, labels_) #sigmoid at the end
                if OPTIMIZE_LOSS:
                    losses.append(loss.item())
                else:
                    outputs = torch.round(torch.sigmoid(outputs))
                    prediction_numpy = outputs.cpu().numpy()
                    assert not np.isnan(prediction_numpy).any(), f'epoch {epoch}, batch_idx {batch_idx}, a prediction is a NaN'
                    predictions.extend(prediction_numpy)
                    labels.extend(labels_.cpu().numpy())

        if OPTIMIZE_LOSS:
            loss_fold.append(np.mean(np.array(losses)))
            trial.report(loss_fold[-1],  fold)
        else:
            metric_fold.append(compute_metric(np.array(labels), np.array(predictions), EVAL_METRIC)) 
            trial.report(metric_fold[-1],  fold)
            # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    if OPTIMIZE_LOSS:
        mean_loss = np.mean(np.array(loss_fold))
        #print(f'Mean Loss: {mean_loss:.3f}')
        #print('=======\n')
        return mean_loss
    else:
        mean_metric = np.mean(np.array(metric_fold))
        #print(f'Mean {EVAL_METRIC}: {mean_metric*100:.2f}%')
        #print('=======\n')
        return mean_metric


if __name__ == "__main__":
   #augmentation_type = args.augmentation_type
    center = 'bern'
    hemisphere = 'flipped'
    #resolution = 250

    timestamp = datetime.now().strftime('%Y.%m.%d_%H:%M:%S')
    n_trials = 100

    if OPTIMIZE_LOSS:
        study_name = f'VTANet_Opti2_Loss_{n_trials}_trials__{timestamp}'
        direction = 'minimize'
        #pruner = BasePruner()
    else : 
        study_name = f'VTANet_Opti2_{EVAL_METRIC}_{n_trials}_trials__{timestamp}'
        direction = 'maximize'
        #pruner = BasePruner() #ThresholdPruner(lower=0.6)

    study = optuna.create_study(
        direction = direction, 
        study_name = study_name, 
        #pruner = optuna.pruners.PatientPruner(optuna.pruners.MedianPruner(), patience=2),
        pruner = ThresholdPruner(lower=0.4),
        storage='sqlite:///vtanet_opti2.db')
    try:
        study.optimize(objective, n_trials=n_trials)
    except KeyboardInterrupt:
        pass

    df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    path = joinfile(MODELS_DIR, f'Optuna/VTANet/Opti2/{timestamp}.csv')
    Path(path).parents[0].mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False) 

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    #pickle.dump(trial, 'study.pkl')
    #study = pickle.load("study.pkl")