#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys
from sys import exit
import os
import numpy as np
import torch
from models.model_loader import model_loader
from data_loader.wfdb_dataset import DataLoader
from options.test_opt import Opt
from data_loader.data_read_utils import load_record_use_15s

import time

all_time = 0


def time_me(fn):
    def _wrapper(*args, **kwargs):
        global all_time
        start = time.perf_counter()
        fn(*args, **kwargs)
        all_time += time.perf_counter() - start
        print("%s cost %s second : all time %s second" % (fn.__name__, time.perf_counter() - start, all_time))

    return _wrapper


test_people_preds = list()


def load_model(path, opt):
    model = model_loader(opt)
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    model.eval()
    return model


def load_dataset(opt, filename, alarm_type='all', explict_sensor=True):
    dataset = DataLoader(opt=opt, testfile=filename, alarm_type=alarm_type)
    if not dataset.check_file_valid(alarm_type=alarm_type,
                                    explict_sensor=explict_sensor,
                                    need_long_record=False):
        return None
    dataset.load_data()
    return dataset


import itertools


def vote():
    global test_people_preds
    if not test_people_preds:
        return 0
    preds = list(itertools.chain(*test_people_preds))
    n = len(preds)
    p = np.array(preds)[:, 0].sum() / n
    l = 1 if p < 0.5 else 0
    clear()
    return p, l


def eval_on_single_people(inputs, extra, model, cuda=False):
    model.eval()
    with torch.no_grad():
        input_vars = torch.tensor(inputs, requires_grad=False)
        if np.any(extra):
            extra_vars = torch.tensor(extra, requires_grad=False)
            extra_vars = extra_vars.float()
            extra_vars = extra_vars.cuda() if cuda else extra_vars
        if cuda:
            input_vars = input_vars.cuda()

        outputs = model(input_vars) if not np.any(extra) else model(input_vars, extra_vars)
        res = torch.exp(outputs)
        if hasattr(res, "cpu"):
            res = res.cpu().detach().numpy()
        else:
            res = [o.cpu().detach().numpy() for o in res]
    return res


def predict(dataset, model, cuda=False):
    global test_people_preds
    if isinstance(dataset.sig, list):
        for sig in dataset.sig:
            preds = eval_on_single_people(sig, dataset.extra, model, cuda)
            test_people_preds.append(preds)
    else:
        if np.all(dataset.sig) != None:
            preds = eval_on_single_people(dataset.sig, dataset.extra, model, cuda)
            test_people_preds.append(preds)
    return


def predict_testfile(opt, filename, alarm_type='all', explict_sensor=True, base_path=None, fold=5):
    dataset = load_dataset(opt=opt, filename=filename, alarm_type=alarm_type, explict_sensor=explict_sensor)
    if not dataset:
        return
    opt.input_nc = len(opt.load_sensor_names) if not opt.load_important_sig else 1
    opt.input_length = opt.window_size
    if opt.use_extra:
        opt.extra_length = dataset.get_extro_len()
    for i in range(fold):
        model = load_model(path=base_path + str(i + 1) + '/best.pth', opt=opt)
        predict(dataset, model, opt.cuda)
    return


def predict_testfile_valid_mode(opt, filename, alarm_type='all', explict_sensor=True, base_path=None,
                                model_folder='1'):
    dataset = load_dataset(opt=opt, filename=filename, alarm_type=alarm_type, explict_sensor=explict_sensor)
    if not dataset:
        return
    opt.input_nc = len(opt.load_sensor_names) if not opt.load_important_sig else 1
    opt.input_length = opt.window_size
    if opt.use_extra:
        opt.extra_length = dataset.get_extro_len()
    model = load_model(path=base_path + model_folder + '/best.pth', opt=opt)
    predict(dataset, model, opt.cuda)
    return


def predict_combined(opt, filename, alarm_type='all', base_path='./checkpoints/Combined_edgcn_12s_slice_norm/'):
    global test_people_preds
    opt.cuda = False

    opt.load_sensor_names = ['PLETH', 'RESP', 'ABP']
    predict_testfile(opt, filename=filename, alarm_type=alarm_type, explict_sensor=False,
                     base_path=base_path + 'PRA/')

    opt.load_sensor_names = ['II']
    predict_testfile(opt, filename=filename, alarm_type=alarm_type, explict_sensor=True,
                     base_path=base_path + 'II/')
    opt.load_sensor_names = ['V']
    predict_testfile(opt, filename=filename, alarm_type=alarm_type, explict_sensor=True,
                     base_path=base_path + 'V/')

    opt.load_important_sig = True
    predict_testfile(opt, filename=filename, alarm_type=alarm_type, explict_sensor=False,
                     base_path=base_path + 'ECG/')
    opt.load_important_sig = False

    p, ans = vote()
    return p, ans


def predict_combined_valid(opt, filename, alarm_type='all',
                           base_path='./checkpoints/Combined_edgcn_12s_slice_norm_noise/',
                           model_folder='1'):
    global test_people_preds
    opt.cuda = False

    opt.load_sensor_names = ['PLETH', 'RESP', 'ABP']
    predict_testfile_valid_mode(opt, filename=filename, alarm_type=alarm_type, explict_sensor=False,
                                base_path=base_path + 'PRA/', model_folder=model_folder)

    opt.load_sensor_names = ['II']
    predict_testfile_valid_mode(opt, filename=filename, alarm_type=alarm_type, explict_sensor=True,
                                base_path=base_path + 'II/', model_folder=model_folder)
    opt.load_sensor_names = ['V']
    predict_testfile_valid_mode(opt, filename=filename, alarm_type=alarm_type, explict_sensor=True,
                                base_path=base_path + 'V/', model_folder=model_folder)

    opt.load_important_sig = True
    predict_testfile_valid_mode(opt, filename=filename, alarm_type=alarm_type, explict_sensor=False,
                                base_path=base_path + 'ECG/', model_folder=model_folder)
    opt.load_important_sig = False

    p, ans = vote()
    return p, ans


def predict_edgcn_valid(opt, filename, base_path="./checkpoints/cv_edgcn_all_12s_slice_norm/",
                        model_folder='1'):
    global test_people_preds

    opt.load_sensor_names = ['II', 'V', 'RESP', 'PLETH', 'ABP']
    predict_testfile_valid_mode(opt, filename=filename, alarm_type='all', explict_sensor=False,
                                base_path=base_path, model_folder=model_folder)
    p, ans = vote()
    return p, ans


def predict_dgcn_valid(opt, filename, base_path="./checkpoints/cv_dgcn_all_12s_slice_norm/", model_folder='1'):
    global test_people_preds
    opt.load_sensor_names = ['II', 'V', 'RESP', 'PLETH', 'ABP']
    predict_testfile_valid_mode(opt, filename=filename, alarm_type='all', explict_sensor=False,
                                base_path=base_path, model_folder=model_folder)
    p, ans = vote()
    return p, ans


def predict_dgcn(opt, filename):
    global test_people_preds
    base_path = "checkpoints/cv_dgcn_all_12s_slice/"
    opt.load_sensor_names = ['II', 'V', 'RESP', 'PLETH', 'ABP']

    predict_testfile(opt, filename=filename, alarm_type='all', explict_sensor=False,
                     base_path=base_path)
    ans = vote()
    return ans


def predict_edgcn(opt, filename):
    global test_people_preds
    base_path = "./checkpoints/deep_embeded_edgcn_12s_slice/"
    opt.load_sensor_names = ['II', 'V', 'RESP', 'PLETH', 'ABP']
    predict_testfile(opt, filename=filename, alarm_type='all', explict_sensor=False,
                     base_path=base_path)
    p, ans = vote()
    return p, ans


def clear():
    global test_people_preds
    test_people_preds.clear()
    return


def opt_15s_combined():
    opt = Opt()
    opt.drop_prob = 0.3
    opt.use_minmax_scale = False
    opt.use_global_minmax = False
    opt.model_name = 'deep_embedded'
    opt.load_sensor_names = ['II', 'V', 'RESP', 'PLETH', 'ABP']
    opt.window_size = 3750
    opt.use_extra = True
    opt.add_noise_prob = 0
    opt.cuda = False
    return opt


def opt_12s_norm_combined():
    opt = Opt()
    opt.drop_prob = 0.3
    opt.use_minmax_scale = True
    opt.use_global_minmax = False
    opt.model_name = 'deep_embedded'
    opt.load_sensor_names = ['II', 'V', 'RESP', 'PLETH', 'ABP']
    opt.window_size = 3000
    opt.use_extra = True
    opt.add_noise_prob = 0
    opt.cuda = False
    return opt


if __name__ == "__main__":

    # opt = Opt()
    # opt.drop_prob = 0.3
    # opt.use_minmax_scale = True
    # opt.use_global_minmax = False
    # opt.model_name = 'deep_embedded'
    # opt.load_sensor_names = ['II', 'V', 'RESP', 'PLETH', 'ABP']
    # opt.window_size = 3000
    # opt.use_extra = True
    # opt.add_noise_prob = 0
    # opt.cuda = False

    with open("answers.txt", "a+", encoding="utf-8") as fp:
        for record in sys.argv[1:]:
            if load_record_use_15s(record):
                opt = opt_15s_combined()
                base_path = './checkpoints/Combined_edgcn_15s/'
            else:
                opt = opt_12s_norm_combined()
                base_path = './checkpoints/Combined_edgcn_12s_slice_norm/'
            output_file = os.path.basename(record)
            p, results = predict_combined(opt, record, base_path=base_path)
            fp.write(output_file + ',' + str(results) + "\n")

    sys.exit(0)
