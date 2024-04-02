import time
import numpy as np

class ModelTimer():
    def __init__(self, print_epoch : bool = False):
        self.print_epoch = print_epoch
        self.reset()
        self.fold = 0
        self.timer_dict = {}

# ---- FOLD --------------------------------------------------------

    def new_fold(self):
        self.fold += 1
        self.start_fold_time = time.perf_counter()

    def end_fold(self):
        self.stop_fold_time = time.perf_counter()
        self.fold_timer = self.stop_fold_time - self.start_fold_time
        self.timer_dict[f'Fold {self.fold}, Total Runtime'] = self.fold_timer
        for i, epoch in enumerate(self.epochs):
            self.timer_dict[f'Fold {self.fold}, Epoch {i}'] = epoch
        self.timer_dict[f'Fold {self.fold}, Epoch Average'] = self.average_epoch_time()
        self.timer_dict[f'Fold {self.fold}, Training Runtime'] = self.train_timer
        self.timer_dict[f'Fold {self.fold}, Testing Runtime'] = self.test_timer
        self.reset()

    def print_fold(self):
        print(f'Fold Runtime : {self.fold_timer:.2f}s')

# ---- TRAIN --------------------------------------------------------

    def start_train(self):
        self.start_train_time = time.perf_counter()

    def stop_train(self):
        self.stop_train_time = time.perf_counter()
        self.train_timer = self.stop_train_time - self.start_train_time

    def get_train_time(self):
        return self.train_timer

    def print_train(self):
        print(f'Training time : {self.train_timer:.2f}s')
        print(f'Average time per epoch (total : {len(self.epochs)}): {self.average_epoch_time():.2f}s')

# ---- TEST --------------------------------------------------------

    def start_test(self):
        self.start_test_time = time.perf_counter()

    def stop_test(self):
        if self.start_test_time == np.nan:
            raise RuntimeError('Start test function has not been called before!')

        self.stop_test_time = time.perf_counter()
        self.test_timer = self.stop_test_time - self.start_test_time

    def get_test_time(self):
        return self.test_timer
    
    def print_test(self):
        print(f'Testing time : {self.test_timer:.2f}s')
    

# ---- RAND --------------------------------------------------------

    def start(self):
        self.start_rand_time = time.perf_counter()

    def stop(self):
        self.stop_rand_time = time.perf_counter()
        self.rand_timer = self.stop_rand_time - self.start_rand_time

    def get_time(self):
        return self.rand_timer

# ---- EPOCH --------------------------------------------------------

    def start_epoch(self):
        self.start_epoch_time = time.perf_counter()

    def stop_epoch(self):
        self.stop_epoch_time = time.perf_counter()
        self.last_epoch_timer = self.stop_epoch_time - self.start_epoch_time
        self.epochs.append(self.last_epoch_timer)
        if self.print_epoch:
            print(f'Epoch {len(self.epochs)} : {self.last_epoch_timer:.2f}s')

    def average_epoch_time(self):
        return np.mean(self.epochs)

# ---- MISC --------------------------------------------------------

    def reset(self):
        self.train_timer_= np.nan
        self.test_timer = np.nan
        self.epochs = []

    def get_dict(self):
        return self.timer_dict
    