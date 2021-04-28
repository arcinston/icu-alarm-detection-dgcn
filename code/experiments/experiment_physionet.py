import logging
import os
from collections import OrderedDict
from copy import deepcopy
import time

import pandas as pd
import torch as th
import numpy as np

from experiments.loggers import PrinterTable, PrinterCSVFile
from experiments.stopcriteria import MaxEpochs, ColumnBelow, Or
from torch_ext.util import np_to_var, var_to_np

log = logging.getLogger(__name__)


class RememberBest(object):

    def __init__(self, column_name, increase=False):
        self.column_name = column_name
        self.best_epoch = 0
        self.highest_val = -float("inf")
        self.lowest_val = float("inf")
        self.model_state_dict = None
        self.optimizer_state_dict = None
        self.increase = increase

    def remember_epoch(self, epochs_df, model, optimizer):
        i_epoch = len(epochs_df) - 1
        current_val = float(epochs_df[self.column_name].iloc[-1])
        if not self.increase:
            if current_val <= self.lowest_val:
                self.best_epoch = i_epoch
                self.lowest_val = current_val
                self.model_state_dict = deepcopy(model.state_dict())
                self.optimizer_state_dict = deepcopy(optimizer.state_dict())
                log.info(
                    "New best {:s}: {:5f}".format(self.column_name, current_val)
                )
                log.info("")
        else:
            if current_val > self.highest_val:
                self.best_epoch = i_epoch
                self.highest_val = current_val
                self.model_state_dict = deepcopy(model.state_dict())
                self.optimizer_state_dict = deepcopy(optimizer.state_dict())
                log.info(
                    "New best {:s}: {:5f}".format(self.column_name, current_val)
                )
                log.info("")

    def reset_to_best_model(self, epochs_df, model, optimizer):
        # Remove epochs past the best one from epochs dataframe
        epochs_df.drop(range(self.best_epoch + 1, len(epochs_df)), inplace=True)
        model.load_state_dict(self.model_state_dict)
        optimizer.load_state_dict(self.optimizer_state_dict)


class Experiment(object):
    def __init__(
            self,
            opt,
            model,
            datasets,
            loss_function,
            optimizer,
            model_constraint,
            monitors,
            stop_criterion,
            remember_best_column,
            save_path='',
            training_increase=False,
            model_loss_function=None,
            batch_modifier=None,
            do_early_stop=True,
            log_0_epoch=True,
            loggers="ALL",
    ):
        self.opt = opt
        self.cross_fold = 0
        self.second_phase = False
        self.con_data = False
        if do_early_stop:
            assert remember_best_column is not None
        self.model = model
        self.datasets = datasets
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.model_constraint = model_constraint
        self.monitors = monitors
        self.stop_criterion = stop_criterion
        self.remember_best_column = remember_best_column
        self.model_loss_function = model_loss_function
        self.batch_modifier = batch_modifier
        self.training_increase = training_increase
        self.save_path = save_path
        self.epochs_df = pd.DataFrame()
        self.before_stop_df = None
        self.rememberer = None
        self.do_early_stop = do_early_stop
        self.log_0_epoch = log_0_epoch
        self.loggers = loggers
        return

    def do_one_exp_run(self):
        log.info("Run until early stop...")
        self.run_until_first_stop()
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        th.save(self.rememberer.model_state_dict, self.save_path + '/best.pth')

    def run(self):
        self.setup_training()
        self.do_one_exp_run()

    def setup_training(self):
        # reset remember best extension in case you rerun some experiment
        if self.do_early_stop:
            self.rememberer = RememberBest(self.remember_best_column, increase=self.training_increase)
        if self.loggers == "ALL":
            import os
            if not os.path.exists("../../results/LOG/"):
                os.makedirs("../../results/LOG/")
            file_path = "../../results/LOG/" + '-'.join(self.save_path.split('/')[2:])
            with open(file_path + '.txt', 'w+') as settings_file:
                settings_file.writelines(self.opt.__repr__())

            # self.loggers = [PrinterTable(), TensorboardWriter(log_dir="./LOG/"),
            #                 PrinterCSVFile(log_dir="./LOG/", exp_name=self.save_path.split('/')[-1]),
            #                 PrinterTableFile(log_dir="./LOG/", exp_name=self.save_path.split('/')[-1])]

            self.loggers = [PrinterTable(file_path=file_path + '.txt'),
                            PrinterCSVFile(file_path=file_path + '.csv'), ]
        self.epochs_df = pd.DataFrame()
        if self.opt.cuda:
            assert th.cuda.is_available(), "Cuda not available"
            self.model.cuda()
        return

    def run_until_first_stop(self):
        self.run_until_stop(remember_best=self.do_early_stop)
        return

    def run_until_stop(self, remember_best):
        if self.log_0_epoch:
            self.monitor_epoch()
            self.log_epoch()
            if remember_best:
                self.rememberer.remember_epoch(
                    self.epochs_df, self.model, self.optimizer
                )

        while not self.stop_criterion.should_stop(self.epochs_df):
            self.run_one_epoch(remember_best)
        return

    def run_one_epoch(self, remember_best):
        start_train_epoch_time = time.time()
        data_files_list = self.datasets.leave_out_files
        files = data_files_list['train']
        batch_generator = self.datasets.get_batches(files, True, 'train')
        for inputs, extra, targets in batch_generator:
            self.train_batch(inputs, extra, targets)
        end_train_epoch_time = time.time()
        log.info(
            "Time only for training updates: {:.2f}s".format(
                end_train_epoch_time - start_train_epoch_time
            )
        )

        self.monitor_epoch()
        self.log_epoch()
        if remember_best:
            self.rememberer.remember_epoch(
                self.epochs_df, self.model, self.optimizer
            )
        return

    def train_batch(self, inputs, extra, targets):
        self.model.train()
        input_vars = np_to_var(inputs)
        if self.opt.use_extra:
            extra_vars = np_to_var(extra)
            extra_vars = extra_vars.float()
            extra_vars = extra_vars.cuda() if self.opt.cuda else extra_vars

        target_vars = np_to_var(targets)
        target_vars = target_vars.long()
        if self.opt.cuda:
            input_vars = input_vars.cuda()
            target_vars = target_vars.cuda()

        self.optimizer.zero_grad()
        if self.opt.use_extra:
            outputs = self.model(input_vars, extra_vars)
        else:
            outputs = self.model(input_vars)
        outputs = th.exp(outputs)
        loss = self.loss_function(outputs, target_vars)
        if self.model_loss_function is not None:
            loss = loss + self.model_loss_function(self.model)
        loss.backward()
        self.optimizer.step()
        if self.model_constraint is not None:
            self.model_constraint.apply(self.model)
        return

    def eval_on_batch(self, inputs, extra, targets):
        self.model.eval()
        with th.no_grad():
            input_vars = np_to_var(inputs)
            target_vars = np_to_var(targets)
            target_vars = target_vars.long()
            if self.opt.use_extra:
                extra_vars = np_to_var(extra)
                extra_vars = extra_vars.float()
                extra_vars = extra_vars.cuda() if self.opt.cuda else extra_vars
            if self.opt.cuda:
                input_vars = input_vars.cuda()
                target_vars = target_vars.cuda()

            outputs = self.model(input_vars) if not self.opt.use_extra else self.model(input_vars, extra_vars)
            res = th.exp(outputs)
            # res = outputs
            loss = self.loss_function(outputs, target_vars)
            if hasattr(res, "cpu"):
                res = res.cpu().detach().numpy()
            else:
                res = [o.cpu().detach().numpy() for o in res]
            loss = loss.cpu().detach().numpy()
        return res, loss

    def voting(self, preds, p=0.5):
        false_p = np.mean(preds)
        if false_p > p:
            return 0
        return 1

    def monitor_epoch(self):
        # 结果
        result_dicts_per_monitor = OrderedDict()
        for m in self.monitors:
            result_dicts_per_monitor[m] = OrderedDict()
            result_dict = m.monitor_epoch()
            if result_dict is not None:
                result_dicts_per_monitor[m].update(result_dict)
        data_files_list = self.datasets.leave_out_files
        for setname, files in data_files_list.items():
            assert setname in ["train", "valid", "test"]
            shuffle = True if setname != 'test' else False
            batch_generator = self.datasets.get_batches(files, shuffle, setname)

            all_preds, all_targets, all_targets_alarm_type = [], [], []
            all_losses, all_batch_sizes = [], []

            for inputs, extra, targets in batch_generator:
                # print(inputs.shape)
                preds, loss = self.eval_on_batch(inputs, extra, targets)
                # 保存结果
                all_losses.append(loss)
                all_batch_sizes.append(len(targets))
                all_preds.append(preds)
                all_targets.append(targets)
                all_targets_alarm_type.append(extra)

            all_batch_sizes = [all_batch_sizes]
            all_losses = [all_losses]

            # 监视器打印输出
            for m in self.monitors:
                result_dict = m.monitor_set(
                    setname,
                    all_preds,
                    all_losses,
                    all_batch_sizes,
                    all_targets,
                    all_targets_alarm_type
                )
                if result_dict is not None:
                    result_dicts_per_monitor[m].update(result_dict)

        row_dict = OrderedDict()
        for m in self.monitors:
            row_dict.update(result_dicts_per_monitor[m])
        self.epochs_df = self.epochs_df.append(row_dict, ignore_index=True)
        assert set(self.epochs_df.columns) == set(row_dict.keys()), (
            "Columns of dataframe: {:s}\n and keys of dict {:s} not same"
        ).format(str(set(self.epochs_df.columns)), str(set(row_dict.keys())))
        # self.epochs_df = self.epochs_df[list(row_dict.keys())]
        # # 保存网络中的参数, 速度快，占空间少
        # th.save(self.model.state_dict(), self.save_path + '/' + str(len(self.epochs_df) - 1) + '.pth')
        return

    def log_epoch(self):
        for logger in self.loggers:
            logger.log_epoch(self.epochs_df)
