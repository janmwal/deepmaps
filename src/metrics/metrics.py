from enum import Enum
from os import read
import collections
import pandas as pd
import numpy as np
import math
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, roc_auc_score,
precision_score, recall_score, f1_score, confusion_matrix, roc_curve, matthews_corrcoef, auc, RocCurveDisplay)
from sklearn.preprocessing import binarize
from src.misc.modeltimer import ModelTimer
import src.misc.joinfile as jf
import torch
import pickle

class Metrics(Enum):
    TN = 'True Negative'
    TP = 'True Positive'
    FN = 'False Negative'
    FP = 'False Positive'
    TOTAL = 'Total Samples'
    ACCURACY = 'Accuracy'
    BAL_ACCURACY = 'Balanced Accuracy'
    PRECISION = 'Precision'
    PREVALENCE = 'Prevalence'
    F1_SCORE = 'F1 Score'
    SENSITIVITY = 'Sensitivity'
    SPECIFICITY = 'Specificity'
    MISS_RATE = 'Miss Rate'
    FALL_OUT = 'Fall-out'
    MCC = 'MCC'


def create_dict_dict_list():
    return collections.defaultdict(create_dict_list)

def create_dict_list():
    return collections.defaultdict(list)

# Source : https://stackoverflow.com/questions/26248654/how-to-return-0-with-divide-by-zero
def div0(a, b, fill=0.0):
    """ a / b, divide by 0 -> `fill`
        div0( [-1, 0, 1], 0, fill=np.nan) -> [nan nan nan]
        div0( 1, 0, fill=np.inf ) -> inf
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide(a,b)
    if np.isscalar(c):
        return c if np.isfinite(c) \
            else fill
    else:
        c[~np.isfinite(c)] = fill
        return c

def compute_multi_threshold_metrics(
    y_true : np.array, 
    y_pred : np.array, 
    thresholds : np.array,
    fold : int = None, 
    epoch : int = None) -> dict:
    #auc = roc_auc_score(y_true, y_pred)
    rows = []
    if thresholds is None:
        fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=1.)
    #print(len(y_true), len(y_pred), len(thresholds))
    #print(thresholds)
    #print(len(np.unique(y_pred)))
    if isinstance(y_pred, list):
        y_pred = np.array(y_pred)
    if y_pred.ndim == 1:   
            y_pred = y_pred.reshape(-1, 1)
    for threshold in thresholds:   
        y_pred_ = binarize(y_pred, threshold=threshold)
        row_dict = return_metrics(y_true, y_pred_)
        if fold is not None:
            row_dict['Fold'] = fold
        if epoch is not None:
            row_dict['Epoch'] = epoch
        row_dict['Threshold'] = threshold
        rows.append(row_dict)
    return rows

def compute_fixed_threshold_metrics(y_true : np.array, y_pred : np.array, fold : int = None, epoch : int = None) -> dict:
    if isinstance(y_pred, list):
        y_pred = np.array(y_pred)
    if y_pred.ndim == 1:   
            y_pred = y_pred.reshape(-1, 1)
    y_pred = binarize(y_pred, threshold=0.5)
    row_dict = return_metrics(y_true, y_pred)
    if fold is not None:
        row_dict['Fold'] = fold
    if epoch is not None:
        row_dict['Epoch'] = epoch
    row_dict['Threshold'] = 0.5
    return row_dict

def get_metrics_list():
    return [metric.value for metric in Metrics]

def return_metrics(y_true : np.array, y_pred : np.array, ) -> dict:
    metrics = {}
    assert len(y_true) == len(y_pred), 'Labels and predictions not the same size'
        
    conf_matx = confusion_matrix(y_true,y_pred) # Row : ground truth [0,1], Columns : Predictions [0,1]
    # TP = VTA is full effect & Network predicts it so
    # TN = VTA is partial effect & Network predicts it so
    # FP = VTA is partial effect & Network predicts a full effect
    # FN = VTA is full effect & Network predicts a partial effect
    tn, fp, fn, tp = conf_matx.ravel()
    total = tn + fp + fn + tp
    tnr = div0(tn,float(tn+fp))
    tpr = div0(tp,float(tp+fn))
    fpr = div0(fp,float(fp+tn))
    fnr = div0(fn,float(fn+tp))
    precision = div0(tp,float(tp+fp))
    accuracy = div0((tp + tn),float(total))
    bal_accuracy = (tpr+tnr)/2.0
    prevalence = div0((fn+tp),float(total))
    f1_score_ = div0((2*tp),(2*tp+fp+fn))

    assert math.isclose(precision,precision_score(y_true, y_pred)), f'Precision metrics not the same across computation \
        methods, check labels and predicitions arrays : {precision} vs {precision_score(y_true, y_pred)}'
    assert math.isclose(accuracy,accuracy_score(y_true, y_pred)), f'Accuracy metrics not the same across computation \
        methods, check labels and predicitions arrays : {accuracy} vs {accuracy_score(y_true, y_pred)}'
    assert math.isclose(bal_accuracy,balanced_accuracy_score(y_true, y_pred)), f'Balanced accuracy metrics not the same across \
        computation methods, check labels and predicitions arrays : {bal_accuracy} vs {balanced_accuracy_score(y_true, y_pred)}'
    assert math.isclose(f1_score_,f1_score(y_true, y_pred)), f'F1_score metrics not the same across \
        computation methods, check labels and predicitions arrays : {f1_score_} vs {f1_score(y_true, y_pred)}'
    assert math.isclose(tpr,recall_score(y_true, y_pred)), f'Recall metrics not the same across \
        computation methods, check labels and predicitions arrays : {tpr} vs {recall_score(y_true, y_pred)}'

    metrics[Metrics.ACCURACY.value] = accuracy
    metrics[Metrics.BAL_ACCURACY.value] = bal_accuracy
    metrics[Metrics.PRECISION.value] = precision
    metrics[Metrics.SENSITIVITY.value] = tpr
    metrics[Metrics.SPECIFICITY.value] = tnr
    metrics[Metrics.F1_SCORE.value] = f1_score_
    metrics[Metrics.PREVALENCE.value] = prevalence
    metrics[Metrics.MISS_RATE.value] = fnr
    metrics[Metrics.FALL_OUT.value] = fpr
    metrics[Metrics.MCC.value] = matthews_corrcoef(y_true, y_pred)
    metrics[Metrics.TN.value] = int(tn)
    metrics[Metrics.TP.value] = int(tp)
    metrics[Metrics.FN.value] = int(fn)
    metrics[Metrics.FP.value] = int(fp)
    metrics[Metrics.TOTAL.value] = int(total)

    return metrics

def get_metric(results : dict, metric : Metrics):
    metric_dict = {}
    for fold, dct in results.items():
        for metric_from_results, val in dct.items():
            if metric_from_results == metric:
                metric_dict[metric] = val
    return metric_dict


def best_model(results : dict, current_fold : int = 1):
    best = True
    best_fold = current_fold
    # results is a double dict [fold][eval_metric]
    maxval = results[current_fold]['auc']
    # Assuming dicts are order
    for fold, dct in results.items():
        for metric, val in dct.items():
            if metric == 'auc' and val > maxval:
                best_fold = fold
                best = False

    return best, best_fold

def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)
    
    return acc

# -----------------------------------------------------------------------------

class ModelPerformance():
    TEST_STR = 'Test'
    VALID_STR = 'Valid'
    TRAIN_STR = 'Train'
    def __init__(self):
        
        # --- Timer ---
        self.timer = ModelTimer()
        self.testing_labels, self.testing_predictions = [], []
        # --- Misc ---
        self.default_eval_metric = Metrics.BAL_ACCURACY
        
        self.loss_epoch = []

    def reset_epoch_results(self):
        self.validation_labels, self.validation_predictions = [], [] # format : rows represents each fold
        self.training_labels, self.training_predictions = [], []

    def print_init(self, name, eval_metric, version):
        print(f'====================== {name} ======================')
        print(f'Evaluation metric : {eval_metric.value}')
        print(f'Model : {name}{version}')

    def add_testing_predictions(self, labels, predictions):
        # done every batch of testing 
        #print(predictions)
        self.testing_labels.extend(labels)
        self.testing_predictions.extend(predictions)     

    def add_training_predictions(self, labels, predictions):
        # done every batch of training      
        self.training_labels.extend(labels)
        self.training_predictions.extend(predictions) 

    def add_validation_predictions(self, labels, predictions):
        # done every batch of training      
        self.validation_labels.extend(labels)
        self.validation_predictions.extend(predictions) 

    def start_epoch(self):
        self.timer.start_epoch()
        self.loss_averaged = []
        self.reset_epoch_results()

    def end_epoch(self):
        # stop epoch timer
        self.timer.stop_epoch()

        # add epoch average loss
        self.loss_epoch.append(self.loss_averaged)

        # append predictions for this epoch
        self.training_predictions_epochs.append(self.training_predictions)
        self.training_labels_epochs.append(self.training_labels)
        self.validation_predictions_epochs.append(self.validation_predictions)
        self.validation_labels_epochs.append(self.validation_labels)

    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
        print("Saved model performance to : %s" % filepath)

    @staticmethod
    def load(filepath):
        with open (filepath, 'rb') as f:
            model_perf = pickle.load(f)
        return model_perf
    
    def reset_fold_results(self):
        self.validation_labels_epochs, self.validation_predictions_epochs = [], []
        self.training_labels_epochs, self.training_predictions_epochs = [], []

    def start_test(self):
        #print('Testing begins')
        self.timer.start_test()

    def stop_test(self):
        self.timer.stop_test()
        #self.timer.print_test()

    def start_train(self):
        self.timer.start_train()

    def stop_train(self):
        self.timer.stop_train()
        #self.timer.print_train()

class ModelPerformanceTrain(ModelPerformance):
    def __init__(self, multi_threshold : bool = False, n_thresholds : int = None):
        super(ModelPerformanceTrain, self).__init__()
        self.multi_threshold = multi_threshold

        if n_thresholds is not None:
            self.thresholds = np.linspace(0, 1, n_thresholds)
        else:
            self.thresholds = None
        super().reset_epoch_results()

        # --- Fold variables ---
        # Pred/Labels [fold, epoch, prediction]
        self.training_predictions_folds = []
        self.training_labels_folds = []
        self.validation_predictions_folds = []
        self.validation_labels_folds = []
        self.testing_predictions_folds = []
        self.testing_labels_folds = []

        self.metrics_list = get_metrics_list()
        self.df_train = pd.DataFrame(columns=['Fold', 'Epoch', 'Threshold', *self.metrics_list])
        self.df_valid = pd.DataFrame(columns=['Fold', 'Epoch', 'Threshold', *self.metrics_list])
        self.df_test = pd.DataFrame(columns=['Fold', 'Threshold', *self.metrics_list])
        self.df_global_metrics = pd.DataFrame(columns=['Type', 'Period', 'Threshold', *self.metrics_list])
        self.df_loss = pd.DataFrame(columns=['Type', 'Fold', 'Epoch', 'Batch', 'Iter', 'Loss'])

        self.loss_fold = []

    def compute_test_metrics(self, fold):

        # if self.multi_threshold:
        #     #self.training_metrics_multi_epochs.append(compute_multi_threshold_metrics(self.training_labels,self.training_predictions))
        #     #print(np.unique(self.testing_predictions))
        #     rows = compute_multi_threshold_metrics(self.testing_labels,
        #         self.testing_predictions, 
        #         self.thresholds,
        #         fold)
        #     for row in rows:
        #         new_df = pd.DataFrame(row, index=[0])
        #         self.df_metrics = pd.concat([self.df_metrics, new_df], ignore_index=True)
        #if 0.5 not in self.thresholds or not self.multi_threshold:
        row = compute_fixed_threshold_metrics(self.testing_labels_folds[fold],self.testing_predictions_folds[fold], fold)
        new_df = pd.DataFrame(row, index=[0])
        self.df_test = pd.concat([self.df_test, new_df], ignore_index=True)

    def compute_epoch_metrics(self, fold, epoch):
        # if self.multi_threshold:
        #     #self.training_metrics_multi_epochs.append(compute_multi_threshold_metrics(self.training_labels,self.training_predictions))
        #     #print(np.unique(self.testing_predictions))
        #     rows = compute_multi_threshold_metrics(self.validation_labels,
        #         self.validation_predictions, 
        #         self.thresholds,
        #         fold, 
        #         epoch)
        #     for row in rows:
        #         new_df = pd.DataFrame(row, index=[0])
        #         self.df_valid = pd.concat([self.df_metrics, new_df], ignore_index=True)
        #if 0.5 not in self.thresholds or not self.multi_threshold:
        row = compute_fixed_threshold_metrics(y_true=self.training_labels,y_pred=self.training_predictions, fold=fold, epoch=epoch)
        new_df = pd.DataFrame(row, index=[0])
        self.df_train = pd.concat([self.df_train, new_df], ignore_index=True)

        row = compute_fixed_threshold_metrics(y_true=self.validation_labels,y_pred=self.validation_predictions, fold=fold, epoch=epoch)
        new_df = pd.DataFrame(row, index=[0])
        self.df_valid = pd.concat([self.df_valid, new_df], ignore_index=True)


    def compute_overall_metrics(self):

        pass

        # --- COMPUTE TIME INDEPENDANT METRICS ---
        # get all testing predictions and labels
        # all_predictions, all_labels = [], []

        # for fold in range(len(self.testing_predictions_folds)):
        #     for epoch in range(len(self.testing_predictions_folds[fold])):
        #         all_predictions.extend(self.testing_predictions_folds[fold][epoch])
        #         all_labels.extend(self.testing_labels_folds[fold][epoch])
        
        # if self.multi_threshold:
        #     rows = compute_multi_threshold_metrics(
        #             all_labels, 
        #             all_predictions, 
        #             self.thresholds,
        #             self.TEST_STR)
        #     for row in rows:
        #         row['Period'] = 'All Epochs'
        #         new_df = pd.DataFrame(row, index=[0])
        #         self.df_global_metrics = pd.concat([self.df_global_metrics, new_df], ignore_index=True)
        # if 0.5 not in self.thresholds or not self.multi_threshold:
        #     row = compute_fixed_threshold_metrics(all_labels, all_predictions, self.TEST_STR)
        #     row['Period'] = 'All Epochs'
        #     new_df = pd.DataFrame(row, index=[0])
        #     self.df_global_metrics = pd.concat([self.df_global_metrics, new_df], ignore_index=True)
        
        # last_predictions, last_labels = [], []
        # for fold in range(len(self.testing_predictions_folds)):
        #     last_predictions.extend(self.testing_predictions_folds[fold][-1])
        #     last_labels.extend(self.testing_labels_folds[fold][-1])
        
        # if self.multi_threshold:
        #     rows = compute_multi_threshold_metrics(
        #             last_labels, 
        #             last_predictions, 
        #             self.thresholds,
        #             self.TEST_STR)
        #     for row in rows:
        #         row['Period'] = 'Last Epoch'
        #         new_df = pd.DataFrame(row, index=[0])
        #         self.df_global_metrics = pd.concat([self.df_global_metrics, new_df], ignore_index=True)
        # if 0.5 not in self.thresholds or not self.multi_threshold:
        #     row = compute_fixed_threshold_metrics(last_labels, last_predictions, self.TEST_STR)
        #     row['Period'] = 'Last Epoch'
        #     new_df = pd.DataFrame(row, index=[0])
        #     self.df_global_metrics = pd.concat([self.df_global_metrics, new_df], ignore_index=True)
        
        #self.df_mean_metrics = self.df_metrics.groupby(['Type', 'Epoch', 'Threshold']).mean().reset_index()
        
        #self.df_metrics.sort_values(['Type', 'Fold', 'Epoch', 'Threshold'], ascending=True, inplace=True)
        #self.df_global_metrics.sort_values(['Type', 'Period', 'Threshold'], ascending=True, inplace=True)
        
    def new_fold(self, fold):
        self.timer.new_fold()
        super().reset_epoch_results()
        self.reset_fold_results()

    def end_fold(self, fold):
        # end fold timer
        self.timer.end_fold()

        self.training_predictions_folds.append(self.training_predictions_epochs)
        self.training_labels_folds.append(self.training_labels_epochs)
        self.validation_predictions_folds.append(self.validation_predictions_epochs)
        self.validation_labels_folds.append(self.validation_labels_epochs)
        self.testing_predictions_folds.append(self.testing_predictions)
        self.testing_labels_folds.append(self.testing_labels)
        self.testing_predictions = []
        self.testing_labels = []
        #print(f' length : {len(self.testing_labels_folds)}, last vect len : {len(self.testing_labels_folds[-1])}')
        # add loss for this epoch
        self.loss_fold.append(self.loss_epoch)

        self.compute_test_metrics(fold)


    def reset_fold_results(self):
        super().reset_fold_results()
        # Mutli-threshold Metrics [epoch, threshold, metric]
        self.training_metrics_multi_epochs, self.testing_metrics_multi_epochs = [], []
        # Fixed-threshold Metrics [epoch, metric]
        self.training_metrics_fixed_epochs, self.testing_metrics_fixed_epochs = [], []

    def end_epoch(self, fold, epoch):
        super().end_epoch()

        # compute metrics for this epoch
        self.compute_epoch_metrics(fold, epoch)

    def print_current_fold(self, eval_metric, current_fold):
        print(f'{eval_metric.value} in fold {current_fold+1}: '
              f'{100*(self.fold_metrics[-1][eval_metric]):.2f}%')

    def print_train_results(self, folds, metric : Metrics = None):
        # TODO : REWRITE THIS
        if metric is None:
            metric = self.default_eval_metric
        #print('------------------------------------------------------')
        #print(f'{folds}-FOLD CROSS VALIDATION RESULTS')
        #print(f'Mean ROC-AUC from each fold last epoch : {self.last_testing_metrics_multi["auc"]:.4f}')

    def add_training_loss(self, fold, epoch, batch, iter, loss):
        # Adding averaged loss per example
        new_df = pd.DataFrame(data={'Type' : 'Train', 'Fold' : fold, 'Epoch' : epoch, 'Batch' : batch, 'Iter' : iter, 'Loss' : loss}, index=[0])
        self.df_loss = pd.concat([self.df_loss, new_df], ignore_index=True)

    def add_testing_loss(self, fold, batch, iter, loss):
        # Adding averaged loss per example
        new_df = pd.DataFrame(data={'Type' : 'Test', 'Fold' : fold, 'Epoch' : np.nan, 'Batch' : batch, 'Iter' : iter, 'Loss' : loss}, index=[0])
        self.df_loss = pd.concat([self.df_loss, new_df], ignore_index=True)

    def add_validation_loss(self, fold, epoch, batch, iter, loss):
        # Adding averaged loss per example
        new_df = pd.DataFrame(data={'Type' : 'Valid', 'Fold' : fold, 'Epoch' : epoch, 'Batch' : batch, 'Iter' : iter, 'Loss' : loss}, index=[0])
        self.df_loss = pd.concat([self.df_loss, new_df], ignore_index=True)

    def is_best_model(self, current_fold : int = 0):
        # TODO : REWRITE THIS
        best = True
        maxval = self.testing_metrics_epochs[current_fold]['auc']
        for fold in range(len(self.testing_metrics_epochs)):
            for metric, val in self.testing_metrics_epochs[fold].items():
                if metric == 'auc' and val > maxval:
                    best = False
        return best

    def get_best_fold(self, eval_metric : Metrics = None):
        # TODO : REWRITE THIS

        if eval_metric is None:
            eval_metric = self.default_eval_metric
        maxval = self.fold_metrics[0][metric]
        for fold in range(len(self.fold_metrics)):
            for metric, val in self.fold_metrics[fold].items():
                if metric == eval_metric and val > maxval:
                    best_fold = fold
        return best_fold
    
class ModelPerformancePredict(ModelPerformance):
    def __init__(self):
        super(ModelPerformancePredict, self).__init__()
        #patient specific
        self.training_predictions_patients = []
        self.training_labels_patients = []

        self.testing_predictions_patients = []
        self.testing_labels_patients = []

        self.training_indices_patients = []
        self.testing_indices_patients = []
        #epoch
        self.training_indices_epochs = []
        self.testing_indices_epochs = []
        # indices
        self.training_indices = []
        self.testing_indices = []

        self.reset_patient_results()
    
    def print_init(self, name, version):
        print(f'====================== {name} ======================')
        print(f'Model : {name}{version}')

    def add_testing_predictions(self, labels, predictions, indices):
        # done every batch of testing 
        #print(predictions)
        self.testing_labels.extend(labels)
        self.testing_predictions.extend(predictions) 
        self.testing_indices.extend(indices)    

    def add_training_predictions(self, labels, predictions, indices):
        # done every batch of training      
        self.training_labels.extend(labels)
        self.training_predictions.extend(predictions) 
        self.training_indices.extend(indices)

    def end_epoch(self, fold, epoch):
        super().end_epoch()

        self.training_indices_epochs.append(self.training_indices)
        self.testing_indices_epochs.append(self.testing_indices)

        self.training_indices = []
        self.testing_indices = []

    def reset_patient_results(self):
        super().reset_fold_results()

        self.testing_labels_epochs, self.testing_predictions_epochs = [], []
        self.training_labels_epochs, self.training_predictions_epochs = [], []
        self.training_indices_epochs, self.testing_indices_epochs = [], []

    def end_patient(self, patient, epoch):
        self.training_predictions_patients.append(self.training_predictions_epochs)
        self.testing_predictions_patients.append(self.testing_predictions_epochs)

        self.training_labels_patients.append(self.training_labels_epochs)
        self.testing_labels_patients.append(self.testing_labels_epochs)

        self.training_indices_patients.append(self.training_indices_epochs)
        self.testing_indices_patients.append(self.testing_indices_epochs)

        self.training_indices = []
        self.testing_indices = []

        self.reset_patient_results()


# -----------------------------------------------------------------------------       
def get_accuracy_tensor(y_true, y_prob, threshold = 0.5):
    y_true = torch.squeeze(y_true, 1)
    y_prob = torch.squeeze(y_prob, 1)

    assert (y_true.ndim == 1) and (y_true.size() == y_prob.size()), f'{y_true.size()} {y_prob.size()}'
    y_prob = y_prob > threshold
    return (y_true == y_prob).sum().item() / y_true.size(0)

def get_balanced_accuracy(y_true, y_prob):
    y_true = y_true.squeeze()
    y_prob = y_prob.squeeze()

    return balanced_accuracy_score(y_true, y_prob)

def get_balanced_accuracy_tensor(y_true, y_prob, threshold = 0.5):
    y_true = torch.squeeze(y_true, 1)
    y_prob = torch.squeeze(y_prob, 1)

    assert (y_true.ndim == 1) and (y_true.size() == y_prob.size()), f'{y_true.size()} {y_prob.size()}'

    y_true = y_true.detach().cpu().numpy()
    y_prob = y_prob.detach().cpu().numpy()
    
    return balanced_accuracy_score(y_true, y_prob)