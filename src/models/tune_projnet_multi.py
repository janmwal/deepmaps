import os
import importlib.util
import sys
from datetime import datetime
import argparse
import pickle
import copy
import time

#sys.path.append("/media/brainstimmaps/DATA/2009_DeepMaps01/04_Source/01_Development/deepmaps/")

import optuna
from optuna.trial import TrialState
from optuna.pruners import ThresholdPruner, NopPruner, BasePruner
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

import numpy as np
from tqdm import tqdm
from pathlib import Path
# Project libraries --------------------------------
from src.misc.joinfile import joinfile
from src.models.constants import *
from dir_const import MODELS_DIR, SRC_DATASET_DIR

EPOCHS = 12
MAX_BATCH_SIZE = 1024
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
        raise ValueError("Invalid function")


def define_model_4090(trial, img_shape):
    # We optimize the number of layers, hidden units and dropout ratio in each layer.  
    layers = []
    #print(img_shape)
    # TODO : int is better than categorical in bayesian optimization
    conv_shape = img_shape['in_shape'][1:]

    in_channels = 3
    padding = 1
    verbose = False
    tensor_debug = False

    MAX_CONV_LAYERS = 8
    MAX_FC_LAYERS = 2

    n_conv_layers = trial.suggest_int("n_conv_layers", 1, MAX_CONV_LAYERS)

    if verbose : print(f'{n_conv_layers} conv layers\n')

    conv_shapes = []
    for i in range(MAX_CONV_LAYERS):
        out_channels = trial.suggest_int(
                f"conv_{i}", 
                10,
                150)
        conv_shapes.append(out_channels)
    

    for i in range(n_conv_layers):
        out_channels = conv_shapes[i]

        if verbose: print(f'FC layer {i}\tIn : {in_features}\tOut: {out_features}') 

        layers.append(nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, 
            kernel_size=3, padding=padding))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm2d(out_channels))

        #layers.append(nn.MaxPool3d(2, 2, ceil_mode=True))
        conv_shape = np.array([np.ceil((x-2+2*padding)).astype(np.uint8) for x in conv_shape])

        if verbose: print('\tOut conv shape : {}'.format(conv_shape))

        in_channels = out_channels

        if tensor_debug:
            layers.append(PrintLayer())
    

    layers.append(nn.Flatten())

    product = np.prod(conv_shape)
    in_features = (product*out_channels).astype(np.uint32)
    
    if verbose:
        print('Conv shape : {} = {}'.format(conv_shape, product))
        print('Out channels : {} '.format(out_channels))
        print('\n=======')
        print('Flatten features  : {} x {} = {}'.format(out_channels, product, in_features))
        print('=======\n')

    n_fc_layers = trial.suggest_int("n_fc_layers", 1, MAX_FC_LAYERS)

    fc_shapes = []
    for i in range(MAX_FC_LAYERS):
        out_channels = trial.suggest_int(
                f"fc_{i}", 
                10,
                150)
        fc_shapes.append(out_channels)

    dropout = trial.suggest_float(
                "dropout_fc".format(i), 
                0.1, 0.8)
    
    for i in range(n_fc_layers):
        out_features = fc_shapes[i]

        if verbose: print(f'FC layer {i}\tIn : {in_features}\tOut: {out_features}') 

        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.ReLU())
        
        layers.append(nn.Dropout1d(dropout))
        in_features = out_features
        if tensor_debug:
            layers.append(PrintLayer()) 

    if verbose : print()

    layers.append(nn.Linear(in_features, 1))
    return nn.Sequential(*layers)

# ---

def objective(trial, n_split):

    one_time_user_attr_reporting = True
    # Device configuration  
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #print(torch.cuda.current_device())
    
    # Loading the dataset
    model_name = 'proj_net'
    dataset_module_name = models_meta[model_name]['dataset_module_name']
    dataset_class_name = models_meta[model_name]['dataset_class_name']
    #resolution = trial.suggest_categorical("resolution", [250, 500, 1000])
    dataset_spec = importlib.util.spec_from_file_location(
        dataset_module_name,
        joinfile(SRC_DATASET_DIR, f'{dataset_module_name}.py'))
    dataset_module = importlib.util.module_from_spec(dataset_spec)
    sys.modules[dataset_module_name] = dataset_module
    dataset_spec.loader.exec_module(dataset_module)
    label_threshold = trial.suggest_float("label_threshold", 0.5, 0.999)
    #augmentation_type = trial.suggest_categorical("augmentation_type", ['augmented_part', 'augmented_full'])
    interpolation = trial.suggest_categorical("interpolation", ['linear_interp', 'step_interp'])
    noise_factor = trial.suggest_float("noise_factor", 3.0, 40.0)
    epochs = trial.suggest_int("epoch", 3, 20)

    resolution = trial.suggest_categorical("resolution", [250, 500, 1000])

    dataset = getattr(dataset_module, dataset_class_name)(
        input_folder_vtas=None, 
        input_folder_table=None,
        space='stn_space_3sigma_subtracted_tweened',
        center='merged',
        hemisphere='flipped',
        resolution=resolution,
        augmentation_type='augmented_full',
        interpolation=interpolation,
        label_threshold=label_threshold,
        normalize_projections=False,
        noise_factor=noise_factor,
        n_tuning=n_split,
        tweening=False)
    img_shape = dataset.get_model_args()

    # Generate the optimizers.
    #optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD", "Adamax"])
    optimizer_name = 'Adam'
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    

    chosen_batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512, 1024])
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

    training_times = []

    weightdecay = trial.suggest_float("weight_decay", 1e-3, 2, log=True)
    lambda_coeff = trial.suggest_float("lambda_coeff", 0.4, 1)
    alpha_early_stopping = trial.suggest_float("alpha_gl", 0., 100.)
    for fold, (train_valid_ids, test_ids) in enumerate(tqdm(kfold.split(dataset, dataset.get_y()), 
                                                      colour = 'blue', 
                                                      desc = 'Folds  ', 
                                                      position = 0,
                                                      leave = False,
                                                      total = kfold.get_n_splits())):
        labels = dataset.get_y()[train_valid_ids]
        
        # train, valid, test have the same class balance
        train_ids, valid_ids = split_train_valid(train_valid_ids, labels)
        sample_weights = get_sample_weights(dataset, train_ids)

        # Creating PT data samplers and loaders:
        train_sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_ids), replacement=True)
        valid_sampler = SubsetRandomSampler(valid_ids)
        test_sampler = SubsetRandomSampler(test_ids)

        if one_time_user_attr_reporting:
            trial.set_user_attr('# total samples', len(labels))
            trial.set_user_attr('# training samples', len(train_sampler))
            trial.set_user_attr('# validation samples', len(valid_sampler))
            trial.set_user_attr('# testing samples', len(test_sampler))
            one_time_user_attr_reporting = False

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
        model = define_model_4090(trial, img_shape).to(device)    
        criterion_name = models_meta[model_name]['criterion']
        
        optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr, weight_decay=weightdecay)
        criterion = getattr(nn, criterion_name)()
        
        lambda1 = lambda epoch: lambda_coeff ** epoch
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

        # Early stopping
        min_loss = 100.0
        patience = trial.suggest_int("patience", 1, 3)
        triggertimes = 0 
        
        st = time.time()

        # Training of the model.
        for epoch in tqdm(range(epochs), colour = 'cyan', desc = 'Epochs ', position = 1, leave = False):
            
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

        training_times.append((time.time() - st)/ 60.)

        losses = []
        predictions, labels = [], []
        best_model.eval()
        with torch.no_grad():
            for batch_idx, (images, labels_, index) in enumerate(test_loader):
                images = images.to(device).float()
                labels_ = labels_.to(device).float()

                outputs = best_model(images)
                loss = criterion(outputs, labels_) #sigmoid at the end
                
                losses.append(loss.item())
                
                outputs = torch.round(torch.sigmoid(outputs))
                prediction_numpy = outputs.cpu().numpy()
                assert not np.isnan(prediction_numpy).any(), f'epoch {epoch}, batch_idx {batch_idx}, a prediction is a NaN'
                predictions.extend(prediction_numpy)
                labels.extend(labels_.cpu().numpy())

        loss_fold.append(np.mean(np.array(losses)))
        metric_fold.append(compute_metric(np.array(labels), np.array(predictions), EVAL_METRIC)) 

        if OPTIMIZE_LOSS:          
            trial.report(loss_fold[-1],  fold)
            
        else:
            trial.report(metric_fold[-1],  fold)
            
        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    mean_training_time = np.mean(np.array(training_times))
    trial.set_user_attr(f'mean training time [min]', mean_training_time)

    mean_loss = np.mean(np.array(loss_fold))
    mean_metric = np.mean(np.array(metric_fold))

    if OPTIMIZE_LOSS:
        trial.set_user_attr(f'fold-average {EVAL_METRIC}', mean_metric)
        #print(f'Mean Loss: {mean_loss:.3f}')
        #print('=======\n')
        return mean_loss
    else:
        trial.set_user_attr('fold-average BCE loss', mean_loss)
        #print(f'Mean {EVAL_METRIC}: {mean_metric*100:.2f}%')
        #print('=======\n')
        return mean_metric

class DurationLowerPruner(BasePruner):
    def __init__(self, lower, duration_max):
        self.lower = lower
        self.duration_max = duration_max

    def prune(self, study: "optuna.study.Study", trial: "optuna.trial.FrozenTrial") -> bool:
        step = trial.last_step
        latest_value = trial.intermediate_values[step]
        duration = trial.duration
        lower = self.lower
        duration_max = self.duration_max

        if latest_value < lower or duration > duration_max:
            return True
        else:
            return False

class ConvergenceCallback:
    def __init__(self, pruned_threshold: int):
        self.pruned_threshold = pruned_threshold
        self._consequtive_pruned_count = 0
        self.values = []

    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
        trial.user_attrs.get("previous_best_value", None)
        if trial.state == optuna.trial.TrialState.PRUNED:
            self._consequtive_pruned_count += 1
        else:
            self._consequtive_pruned_count = 0

        if self._consequtive_pruned_count >= self.pruned_threshold:
            study.stop()

def tune():
    timestamp = datetime.now().strftime('%Y.%m.%d_%H:%M:%S')
    for n_split in range(1, 11):
        
        study_name = f'5Fold_Loss_ProjNetMerged4_{timestamp}' if OPTIMIZE_LOSS else f'5Fold_ProjNetMerged4_{EVAL_METRIC}_{timestamp}'
        #study_name = f'5Fold_Loss_ProjNetMerged1_16/08/2023_2' if OPTIMIZE_LOSS else f'5Fold_ProjNetMerged1_{EVAL_METRIC}_16/08/2023_2'
        direction = 'minimize' if OPTIMIZE_LOSS else 'maximize'
        pruner = NopPruner() if OPTIMIZE_LOSS else DurationLowerPruner(lower=0.4, duration_max=...) #TODO : change this
        cb = ConvergenceCallback(pruned_threshold=30)
        study = optuna.create_study(
            direction = direction, 
            study_name = study_name, 
            pruner = pruner,
            storage='sqlite:///projnet_multi.db',
            load_if_exists=True)
        try:
            study.optimize(lambda trial: objective(trial, n_split), n_trials=200, gc_after_trial=True, callbacks = [cb])
        except KeyboardInterrupt:
            pass

        df = study.trials_dataframe(attrs=("number", "value", "params", "duration", "state", "user_attrs"))
        path = joinfile(MODELS_DIR, f'Optuna/ProjNet/Merged5/{timestamp}/split{n_split}.csv')
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

if __name__ == "__main__":
    #print(os.environ['PYTHONPATH'])
    tune()
