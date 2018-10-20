# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np


class CallbackList(object):
    """Container abstracting a list of callbacks.
    # Arguments
        callbacks: List of `Callback` instances.
        queue_length: Queue length for keeping
            running statistics over callback execution time.
    """

    def __init__(self, callbacks=None, queue_length=10):
        callbacks = callbacks or []
        self.callbacks = [c for c in callbacks]
        self.queue_length = queue_length

    def append(self, callback):
        self.callbacks.append(callback)

    def set_params(self, params):
        for callback in self.callbacks:
            callback.set_params(params)

    def set_model(self, model):
        for callback in self.callbacks:
            callback.set_model(model)

    def restore_model(self):
        self.model.model.restore_model(self.model)

    def backup_model(self):
        self.model.model.backup_model(self.model)

    def on_epoch_begin(self, epoch, params=None, trainParams=None):
        """Called at the start of an epoch.
        # Arguments
            epoch: integer, index of epoch.
            logs: dictionary of logs.
        """
        params = params or {}
        if trainParams.restart_train:
            for callback in self.callbacks:
                callback.restart_train(params, trainParams)
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, params, trainParams)

    def on_epoch_end(self, epoch, params=None, trainParams=None):
        """Called at the end of an epoch.
        # Arguments
            epoch: integer, index of epoch.
            logs: dictionary of logs.
        """
        params = params or {}
        trainParams = trainParams or {}
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, params, trainParams)

    def on_batch_begin(self, batch, params=None):
        """Called right before processing a batch.
        # Arguments
            batch: integer, index of batch within the current epoch.
            logs: dictionary of logs.
        """
        params = params or {}
        for callback in self.callbacks:
            callback.on_batch_begin(batch, params)


    def on_batch_end(self, batch, params=None):
        """Called at the end of a batch.
        # Arguments
            batch: integer, index of batch within the current epoch.
            logs: dictionary of logs.
        """
        params = params or {}
        for callback in self.callbacks:
            callback.on_batch_end(batch, params)


    def on_train_begin(self, params=None, trainParams=None):
        """Called at the beginning of training.
        # Arguments
            logs: dictionary of logs.
        """
        params = params or {}
        trainParams = trainParams or {}
        for callback in self.callbacks:
            callback.on_train_begin(params, trainParams)

    def on_train_end(self, params=None):
        """Called at the end of training.
        # Arguments
            logs: dictionary of logs.
        """
        params = params or {}
        for callback in self.callbacks:
            callback.on_train_end(params)


    def __iter__(self):
        return iter(self.callbacks)


class Callback(object):
    """Abstract base class used to build new callbacks.
    # Properties
        params: dict. Training parameters
            (eg. verbosity, batch size, number of epochs...).
        model: instance of `keras.models.Model`.
            Reference of the model being trained.
    The `logs` dictionary that callback methods
    take as argument will contain keys for quantities relevant to
    the current batch or epoch.
    Currently, the `.fit()` method of the `Sequential` model class
    will include the following quantities in the `logs` that
    it passes to its callbacks:
        on_epoch_end: logs include `acc` and `loss`, and
            optionally include `val_loss`
            (if validation is enabled in `fit`), and `val_acc`
            (if validation and accuracy monitoring are enabled).
        on_batch_begin: logs include `size`,
            the number of samples in the current batch.
        on_batch_end: logs include `loss`, and optionally `acc`
            (if accuracy monitoring is enabled).
    """

    def __init__(self):
        self.validation_data = None
        self.model = None

    def set_params(self, params):
        self.params = params

    def set_model(self, model):
        self.model = model

    def restore_model(self):
        self.model.model.restore_model(self.model)

    def backup_model(self):
        self.model.model.backup_model(self.model)

    def on_epoch_begin(self, epoch, params=None, trainParams=None):
        pass

    def on_epoch_end(self, epoch, params=None, trainParams=None):
        pass

    def on_batch_begin(self, batch, params=None):
        pass

    def on_batch_end(self, batch, params=None):
        pass

    def on_train_begin(self, params=None, trainParams=None):
        pass

    def on_train_end(self, params=None):
        pass

    def restart_train(self, params=None, trainParams=None):
        pass


class Standard(Callback):

    def __init__(self):
        super(Standard, self).__init__()

    def on_train_begin(self, params=None, trainParams=None):
        trainParams.restart_train = False
        self.restore_model()

    def restart_train(self, params=None, trainParams=None):
        trainParams.restart_train = False


class EarlyStopping(Callback):
    """Stop training when a monitored quantity has stopped improving.
    # Arguments
        monitor: quantity to be monitored.
        min_delta: minimum change in the monitored quantity
            to qualify as an improvement, i.e. an absolute
            change of less than min_delta, will count as no
            improvement.
        patience: number of epochs with no improvement
            after which training will be stopped.
        verbose: verbosity mode.
        mode: one of {auto, min, max}. In `min` mode,
            training will stop when the quantity
            monitored has stopped decreasing; in `max`
            mode it will stop when the quantity
            monitored has stopped increasing; in `auto`
            mode, the direction is automatically inferred
            from the name of the monitored quantity.
        baseline: Baseline value for the monitored quantity to reach.
            Training will stop if the model doesn't show improvement
            over the baseline.
        restore_best_weights: whether to restore model weights from
            the epoch with the best value of the monitored quantity.
            If False, the model weights obtained at the last step of
            training are used.
    """

    def __init__(self,
                 monitor='validErr',
                 min_delta=0,
                 patience=0,
                 verbose=0,
                 mode='auto',
                 baseline=None,
                 restore_best_weights=True,
                 min_value=0.0001):
        super(EarlyStopping, self).__init__()

        self.monitor = monitor
        self.baseline = baseline
        self.patience = patience
        self.verbose = verbose
        self.min_delta = min_delta
        self.wait = 0
        self.stopped_epoch = 0
        self.restore_best_weights = restore_best_weights
        self.best_weights = None
        self.min_value= min_value

        if mode not in ['auto', 'min', 'max']:
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater
        else:
            if 'Acc' in self.monitor:
                self.monitor_op = np.greater
            else:
                self.monitor_op = np.less

        if self.monitor_op == np.greater:
            self.min_delta *= 1
        else:
            self.min_delta *= -1

    def on_train_begin(self, params=None, trainParams=None):
        # Allow instances to be re-used
        self.wait = 0
        self.stopped_epoch = 0
        if self.baseline is not None:
            self.best = self.baseline
        else:
            self.best = np.Inf if self.monitor_op == np.less else -np.Inf

    def on_epoch_end(self, epoch, params=None, trainParams=None):
        current = self.get_monitor_value(params)
        if current is None:
            return

        if self.monitor_op(current - self.min_delta*current, self.best):
            self.best = current
            params.bestEpoch = epoch
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = self.model.model.get_weights(self.model)
            if current <= self.min_value:
                self.wait = self.patience
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                params.curEpoch = np.inf
                if self.restore_best_weights:
                    if self.verbose > 0:
                        print("Restoring model weights from the end of the best epoch")
                    self.model.model.set_weights(self.model, self.best_weights)

    def on_train_end(self, params=None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))

    def restart_train(self, params=None, trainParams=None):
        self.on_train_begin(params, trainParams)

    def get_monitor_value(self, params):
        monitor_value = params.__getattribute__(self.monitor)[-1]
        return monitor_value

class AdaptiveL2(Callback):

    def __init__(self,
                 monitor='validAcc',
                 treshold=[0.95, 0.65],
                 epoch=5,
                 l2_current=0.01,
                 l2_step=0.01,
                 l2_min_step=0.002,
                 mode='auto'):
        super(AdaptiveL2, self).__init__()

        self.monitor = monitor
        self.treshold = treshold
        self.epoch = epoch
        self.l2_current = l2_current
        self.l2_start = l2_current
        self.l2_step = l2_step
        self.l2_min_step = l2_min_step
        self.tresholdSeen = False

        if mode not in ['auto', 'min', 'max']:
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater
        else:
            if 'Acc' in self.monitor:
                self.monitor_op = np.greater
            else:
                self.monitor_op = np.less

    def on_train_begin(self, params=None, trainParams=None):
        trainParams.l2 = self.l2_start
        self.tresholdSeen = False

    def on_epoch_end(self, epoch, params=None, trainParams=None):
        current = self.get_monitor_value(params)
        if current is None:
            return

        if epoch == self.epoch:
            if self.monitor_op(current, self.treshold[0]):
                self.tresholdSeen = True
                self.l2_current += self.l2_step
                trainParams.restart_train = True
            elif self.monitor_op(self.treshold[1], current) and self.tresholdSeen and self.l2_step > self.l2_min_step:
                self.l2_step = self.l2_step / 2
                self.l2_current -= self.l2_step
                trainParams.restart_train = True

    def restart_train(self, params=None, trainParams=None):
        params.curEpoch = 0
        trainParams.l2 = self.l2_current
        self.restore_model()

    def get_monitor_value(self, params):
        monitor_value = params.__getattribute__(self.monitor)[-1]
        return monitor_value


