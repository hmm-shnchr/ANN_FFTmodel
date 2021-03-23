from multilayer_extend_gpu import MultiLayerNetExtend
from make_train_test_dataset import MakeTrainTestDataset
from reshape_merger_tree import ReshapeMergerTree
from normalization import Normalization
from optimizer_gpu import set_optimizer
import matplotlib.pyplot as plt
import numpy as np
import cupy
import copy
import os, sys


class ArtificialNeuralNetwork:
    def __init__(self, input_size, hidden, act_func, weight_init, batch_norm, output_size, lastlayer_identity, loss_func, is_epoch_in_each_mlist = False):
        self.input_size = input_size
        self.hidden, self.act_func, self.weight_init, self.batch_norm = hidden, act_func, weight_init, batch_norm
        self.output_size = output_size
        self.lastlayer_identity = lastlayer_identity
        self.loss_func = loss_func
        self.network_real = None
        self.network_imag = None
        self.is_epoch_in_each_mlist = is_epoch_in_each_mlist
        if self.is_epoch_in_each_mlist:
            self.loss_val_real, self.loss_val_imag = {}, {}
            self.train_acc_real, self.train_acc_imag = {}, {}
            self.test_acc_real, self.test_acc_imag = {}, {}
        else:
            self.loss_val_real, self.loss_val_imag = [], []
            self.train_acc_real, self.train_acc_imag = [], []
            self.test_acc_real, self.test_acc_imag = [], []

    def set_dataset(self, dataset, train_ratio, fft_format, norm_format):
        mlist = dataset.keys()
        MTTD = MakeTrainTestDataset(mlist)
        train, test = MTTD.split(dataset, train_ratio)
        RMT_train, RMT_test = {}, {}
        train_input, train_output = {}, {}
        test_input, test_output = {}, {}
        train_input_, train_output_ = None, None
        test_input_, test_output_ = None, None
        if fft_format == "fft":
            fft = lambda x: np.fft.fft(x)
        elif fft_format == "rfft":
            fft = lambda x: np.fft.rfft(x)
        if self.is_epoch_in_each_mlist:
            Norm_train_input, Norm_train_output = {}, {}
            Norm_test_input, Norm_test_output = {}, {}
            for m_key in mlist:
                Norm_train_input[m_key] = Normalization(norm_format)
                Norm_train_output[m_key] = Normalization(norm_format)
                Norm_test_input[m_key] = Normalization(norm_format)
                Norm_test_output[m_key] = Normalization(norm_format)
        Norm_train_input_ = Normalization(norm_format)
        Norm_train_output_ = Normalization(norm_format)
        Norm_test_input_ = Normalization(norm_format)
        Norm_test_output_ = Normalization(norm_format)
        for m_key in mlist:
            RMT_train[m_key] = ReshapeMergerTree()
            RMT_test[m_key] = ReshapeMergerTree()
            train_input[m_key], train_output[m_key] = RMT_train[m_key].make_dataset(train[m_key], self.input_size, self.output_size)
            test_input[m_key], test_output[m_key] = RMT_test[m_key].make_dataset(test[m_key], self.input_size, self.output_size)
            if train_input_ is None and test_input_ is None:
                train_input_, train_output_ = train_input[m_key], train_output[m_key]
                test_input_, test_output_ = test_input[m_key], test_output[m_key]
            else:
                train_input_ = np.concatenate([train_input_, train_input[m_key]], axis = 0)
                train_output_ = np.concatenate([train_output_, train_output[m_key]], axis = 0)
                test_input_ = np.concatenate([test_input_, test_input[m_key]], axis = 0)
                test_output_ = np.concatenate([test_output_, test_output[m_key]], axis = 0)
            if self.is_epoch_in_each_mlist:
                train_input[m_key] = Norm_train_input[m_key].run(fft(train_input[m_key]))
                train_output[m_key] = Norm_train_output[m_key].run(fft(train_output[m_key]))
                test_input[m_key] = Norm_test_input[m_key].run(fft(test_input[m_key]))
                test_output[m_key] = Norm_test_output[m_key].run(fft(test_output[m_key]))
        
        train_input_ = Norm_train_input_.run(fft(train_input_))
        train_output_ = Norm_train_output_.run(fft(train_output_))
        test_input_ = Norm_test_input_.run(fft(test_input_))
        test_output_ = Norm_test_output_.run(fft(test_output_))
        train_mask_real = (train_output_.real == 0.0)
        train_mask_imag = (train_output_.imag == 0.0)
        train_output_[train_mask_real] += 1e-7
        train_output_[train_mask_imag] += 1e-7j
        if self.is_epoch_in_each_mlist:
            for m_key in mlist:
                train_mask_real = (train_output[m_key].real == 0.0)
                train_mask_imag = (train_output[m_key].imag == 0.0)
                test_mask_real = (test_output[m_key].real == 0.0)
                test_mask_imag = (test_output[m_key].imag == 0.0)
                train_output[m_key][train_mask_real] += 1e-7
                train_output[m_key][train_mask_imag] += 1e-7j
                test_output[m_key][test_mask_real] += 1e-7
                test_output[m_key][test_mask_imag] += 1e-7j
            return train_input_, train_output_, train_input, train_output, test_input, test_output
        else:
            test_mask_real = (test_output_.real == 0.0)
            test_mask_imag = (test_output_.imag == 0.0)
            test_output_[test_mask_real] += 1e-7
            test_output_[test_mask_imag] += 1e-7j
            return train_input_, train_output_, test_input_, test_output_

    def learning(self, dataset, train_ratio, opt, lr, batchsize_denominator, epoch, fft_format, norm_format):
        mlist = dataset.keys()
        if self.is_epoch_in_each_mlist:
            for m_key in mlist:
                self.loss_val_real[m_key], self.loss_val_imag[m_key] = [], []
                self.train_acc_real[m_key], self.train_acc_imag[m_key] = [], []
                self.test_acc_real[m_key], self.test_acc_imag[m_key] = [], []
            train_input_, train_output_, train_input, train_output, test_input, test_output = self.set_dataset(copy.deepcopy(dataset), train_ratio, fft_format, norm_format)
        else:
            train_input_, train_output_, test_input_, test_output_ = self.set_dataset(copy.deepcopy(dataset), train_ratio, fft_format, norm_format)
        print("Make a train/test dataset.")
        self.network_real = MultiLayerNetExtend(train_input_.shape[1], self.hidden, self.act_func, self.weight_init, self.batch_norm, train_output_.shape[1], self.lastlayer_identity, self.loss_func)
        self.network_imag = MultiLayerNetExtend(train_input_.shape[1], self.hidden, self.act_func, self.weight_init, self.batch_norm, train_output_.shape[1], self.lastlayer_identity, self.loss_func)
        learning_rate = float(lr)
        optimizer_real = set_optimizer(opt, learning_rate)
        optimizer_imag = set_optimizer(opt, learning_rate)
        rowsize_train = train_input_.shape[0]
        batch_mask_arange = np.arange(rowsize_train)
        batch_size = int(rowsize_train / batchsize_denominator)
        iter_per_epoch = int(rowsize_train / batch_size)
        iter_num = iter_per_epoch * epoch
        print("Mini-batch size : {}\nIterations per 1epoch : {}\nTotal iterations : {}".format(batch_size, iter_per_epoch, iter_num))
        for i in range(iter_num):
            batch_mask = np.random.choice(batch_mask_arange, batch_size)
            batch_input_real, batch_input_imag = cupy.asarray(train_input_[batch_mask].real), cupy.asarray(train_input_[batch_mask].imag)
            batch_output_real, batch_output_imag = cupy.asarray(train_output_[batch_mask].real), cupy.asarray(train_output_[batch_mask].imag)
            grads_real = self.network_real.gradient(batch_input_real, batch_output_real, is_training = True)
            grads_imag = self.network_imag.gradient(batch_input_imag, batch_output_imag, is_training = True)
            params_network_real = self.network_real.params
            params_network_imag = self.network_imag.params
            optimizer_real.update(params_network_real, grads_real)
            optimizer_imag.update(params_network_imag, grads_imag)
            if i % iter_per_epoch == 0:
                if self.is_epoch_in_each_mlist:
                    for m_key in mlist:
                        loss_val_real = self.network_real.loss(cupy.asarray(train_input[m_key].real), cupy.asarray(train_output[m_key].real), is_training = False)
                        loss_val_imag = self.network_imag.loss(cupy.asarray(train_input[m_key].imag), cupy.asarray(train_output[m_key].imag), is_training = False)
                        self.loss_val_real[m_key].append(loss_val_real)
                        self.loss_val_imag[m_key].append(loss_val_imag)
                        train_acc_real = self.network_real.predict(cupy.asarray(train_input[m_key].real), is_training = False)
                        train_acc_imag = self.network_imag.predict(cupy.asarray(train_input[m_key].imag), is_training = False)
                        self.train_acc_real[m_key].append(train_acc_real)
                        self.train_acc_imag[m_key].append(train_acc_imag)
                        test_acc_real = self.network_real.predict(cupy.asarray(test_input[m_key].real), is_training = False)
                        test_acc_imag = self.network_imag.predict(cupy.asarray(test_input[m_key].imag), is_training = False)
                        self.test_acc_real[m_key].append(test_acc_real)
                        self.test_acc_imag[m_key].append(test_acc_imag)
