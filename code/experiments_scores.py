import logging
import sys
import os
import numpy as np

import torch as th
from torch_ext.util import set_random_seeds

from options.default_opt import Opt
from datasets.wfdb_dataset import WFDB_Loader
import time

from models.model_loader import load_model
import copy
import itertools

from collections import defaultdict
import prettytable as pt
from options.OptionFactory import gen_options
from sklearn.metrics import confusion_matrix, roc_auc_score

log = logging.getLogger(__name__)
log.setLevel('DEBUG')

alarm_map = {0: 'VTA', 1: 'ETC', 2: 'VFB', 3: 'EBR', 4: 'ASY'}


def np_to_var(
        X, requires_grad=False, dtype=None, pin_memory=False, **tensor_kwargs
):
    if not hasattr(X, "__len__"):
        X = [X]
    X = np.asarray(X)
    if dtype is not None:
        X = X.astype(dtype)
    X_tensor = th.tensor(X, requires_grad=requires_grad, **tensor_kwargs)
    if pin_memory:
        X_tensor = X_tensor.pin_memory()
    return X_tensor


def compute_score(tp, fp, tn, fn):
    acc = (tn + tp) / (tp + tn + fp + fn)
    tpr = tp / max(1, (tp + fn))
    tnr = tn / max(1, (tn + fp))
    ref_score = (tp + fn) / (tp + tn + fp + fn)
    score = (tp + tn) / (tp + tn + fp + 5 * fn)
    return round(acc, 4), round(tpr, 4), round(tnr, 4), round(ref_score, 4), round(score, 4)


def confuse_results(test_people_preds):
    test_people_preds_new = defaultdict(float)
    for f, preds in test_people_preds.items():
        tmp_preds = list(itertools.chain(*preds))
        n = len(tmp_preds)
        p = np.array(tmp_preds)[:, 1].sum() / n
        test_people_preds_new[f] = p
    return test_people_preds_new


def vote(test_people_preds):
    test_people_preds_new = defaultdict(float)
    test_people_result = defaultdict(int)
    for f, preds in test_people_preds.items():
        tmp_preds = list(itertools.chain(*preds))
        n = len(tmp_preds)
        p = np.array(tmp_preds)[:, 0].sum() / n
        test_people_preds_new[f] = p
        l = 1 if p < 0.5 else 0
        test_people_result[f] = l
    return test_people_result, test_people_preds_new


def get_alarm_samples(preds, targets, alarm_types):
    alarm_preds = {'ALL': []}
    alarm_targets = {'ALL': []}
    for i, alarm in enumerate(alarm_types):
        if alarm_map[alarm] not in alarm_preds:
            alarm_preds[alarm_map[alarm]] = []
            alarm_targets[alarm_map[alarm]] = []
        alarm_preds[alarm_map[alarm]].append(preds[i])
        alarm_targets[alarm_map[alarm]].append(targets[i])
        alarm_preds['ALL'].append(preds[i])
        alarm_targets['ALL'].append(targets[i])
    return alarm_preds, alarm_targets


def get_alarm_confusion_matrix(preds, targets, alarm_types):
    alarm_cm = {'ALL': [], 'VTA': [], 'ETC': [],
                'VFB': [], 'EBR': [],
                'ASY': []}
    info = confusion_matrix(targets, preds, labels=[0, 1]).ravel()
    alarm_cm['ALL'] += info
    alarm_map = {0: 'VTA', 1: 'ETC', 2: 'VFB', 3: 'EBR', 4: 'ASY'}
    alarm_preds = {}
    alarm_targets = {}

    for i, alarm in enumerate(alarm_types):
        if alarm_map[alarm] not in alarm_preds:
            alarm_preds[alarm_map[alarm]] = []
            alarm_targets[alarm_map[alarm]] = []
        alarm_preds[alarm_map[alarm]].append(preds[i])
        alarm_targets[alarm_map[alarm]].append(targets[i])

    for alarm, preds in alarm_preds.items():
        info = confusion_matrix(targets, preds, labels=[0, 1]).ravel()
        alarm_cm[alarm] += info
    return alarm_cm


def score(preds, targets, alarm_types):
    alarm_preds, alarm_targets = get_alarm_samples(preds, targets, alarm_types)

    all_scores = []
    for alarm, preds in alarm_preds.items():
        preds = np.array(preds)
        t = alarm_targets[alarm]
        try:
            auc = roc_auc_score(t, preds)
        except ValueError:
            auc = 0
        p = np.int32(preds > 0.5)
        tn, fp, fn, tp = confusion_matrix(t, p, labels=[0, 1]).ravel()
        acc, tpr, tnr, ref_score, score = compute_score(tp, fp, tn, fn)
        all_scores.append(
            [alarm, tn + fp + fn + tp, tp, tn, fp, fn, round(acc, 4), round(auc, 4), round(tpr, 4), round(tnr, 4),
             round(ref_score, 4), round(score, 4)])
    return all_scores


def load_dataset(opt, alarm_type='all', explict_sensor=True):
    log.info("Loading dataset...")
    dataset = WFDB_Loader(opt)
    dataset.filter_files(alarm_type, explict_sensor)
    return dataset


def load_model_p(opt=None, exist_model_path=None):
    assert exist_model_path is not None
    model = load_model(opt)
    model = load_model_para(model, exist_model_path)
    if model is None:
        return None
    model.eval()
    return model


def load_model_para(model, path):
    import torch
    if not os.path.exists(path):
        return None
    model.load_state_dict(torch.load(path))
    return model


def eval_on_single_people(inputs, extra, model, cuda=False):
    model.eval()
    with th.no_grad():
        input_vars = np_to_var(inputs)
        if np.any(extra):
            extra_vars = np_to_var(extra)
            extra_vars = extra_vars.float()
            extra_vars = extra_vars.cuda() if cuda else extra_vars
        if cuda:
            input_vars = input_vars.cuda()
        outputs = model(input_vars) if not np.any(
            extra) else model(input_vars, extra_vars)
        res = th.exp(outputs)
        if hasattr(res, "cpu"):
            res = res.cpu().detach().numpy()
        else:
            # assume it is iterable
            res = [o.cpu().detach().numpy() for o in res]
    return res


def single_exp(opt, explict_sensor=False, alarm_type='all', exist_model_path=None, test_people_preds=None,
               test_people_labels=None, test_people_alarm=None):
    dataset = load_dataset(opt, alarm_type, explict_sensor=explict_sensor)

    opt.n_classes = 2
    opt.input_nc = len(
        opt.load_sensor_names) if not opt.load_important_sig else 1

    opt.input_length = opt.window_size
    if opt.use_extra:
        opt.extra_length = dataset.get_extro_len()
    model = load_model_p(opt, exist_model_path)
    if model is None:
        return None, None, None
    return predict(opt, dataset, model, alarm_type, opt.cuda, test_people_preds=test_people_preds,
                   test_people_labels=test_people_labels, test_people_alarm=test_people_alarm)


def predict(opt, datasets, model, alarm_type='all', cuda=False, test_people_preds=None, test_people_labels=None,
            test_people_alarm=None):
    if not test_people_labels:
        test_people_preds = defaultdict(list)
        test_people_labels = defaultdict(int)
        test_people_alarm = defaultdict(int)
    for file in datasets.need_test_files:
        inputs, extra, targets = datasets.get_people(
            file, alarm_type=alarm_type, check=True)

        if np.all(inputs) is None:
            continue
        if not opt.use_extra:
            extra = None
        if isinstance(inputs, list):
            for sig in inputs:
                preds = eval_on_single_people(sig, extra, model, cuda)
                test_people_preds[file].append(preds[:])
        else:
            preds = eval_on_single_people(inputs, extra, model, cuda)
            test_people_preds[file].append(preds[:])
        test_people_labels[file] = targets[-1]
        test_people_alarm[file] = np.argmax(extra)

    return test_people_preds, test_people_labels, test_people_alarm


def single_combined_element_exp(model_path, signame, opt, explict_sensor=True, alarm_type='all'):
    if isinstance(signame, list):
        opt.load_sensor_names = signame
    else:
        opt.load_sensor_names = [signame]

    test_people_preds, test_people_labels, test_people_alarm = single_exp(opt, explict_sensor=explict_sensor,
                                                                          alarm_type=alarm_type,
                                                                          exist_model_path=model_path)
    if test_people_labels is None:
        return None
    test_people_preds = confuse_results(test_people_preds)
    list_preds, list_labels, list_alarm = results_dict_to_list(
        test_people_preds, test_people_labels, test_people_alarm)
    res = score(list_preds, list_labels, list_alarm)
    log.info("Single element exp: got res")
    return res


def results_dict_to_list(preds: dict, labels: dict, alarm_type: dict):
    list_preds = []
    list_labels = []
    list_alarm = []
    for f, p in preds.items():
        list_preds.append(p)
        list_labels.append(labels[f])
        list_alarm.append(alarm_type[f])
    return list_preds, list_labels, list_alarm


def all_res_log(all_res, exp_name):
    tb = pt.PrettyTable()
    tb.field_names = ['Set', 'Alarm', 'Nums', 'TP', 'TN',
                      'FP', 'FN', 'ACC', 'TPR', 'TNR', 'Ref_score', 'Score']
    with open('../results/'+exp_name.replace('/', '_') + '_test_.txt', 'a+') as log_file:
        for alarm, value in all_res.items():
            stb = pt.PrettyTable()
            stb.field_names = ['Fold', 'Alarm', 'Nums', 'TP', 'TN', 'FP', 'FN', 'ACC', 'AUC', 'TPR', 'TNR', 'Ref_score',
                               'Score']
            for j in range(len(value)):
                stb.add_row([j + 1, alarm] + value[j])
            print(stb)
            log_file.write(str(stb) + '\n')

            tmp = np.array(value)[:, 0:5].astype(np.int)
            tmp = tmp.sum(axis=0)
            a, tp, tn, fp, fn = tmp[0], tmp[1], tmp[2], tmp[3], tmp[4]
            acc, tpr, tnr, ref_score, sc = compute_score(tp, fp, tn, fn)
            tb.add_row([exp_name, alarm, a, tp, tn, fp,
                       fn, acc, tpr, tnr, ref_score, sc])
            time_str = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

        print(tb)
        log_file.write(time_str + ': combined model all result \n')
        log_file.write(str(tb) + '\n')
        log_file.writelines('\n')
    return


def run_cross_val_exp(opt, alarm_type='all', explict_sensor=True, exp_name='all', model_name='best.pth'):
    all_res = {}
    for i in range(5):
        opt.train_files = opt.data_folder + opt.train_datapaths[i]
        opt.test_files = opt.data_folder + opt.test_datapaths[i]
        ename = exp_name + "/" + str(i + 1) + '/'
        model_path = '../results/checkpoints/' + ename + model_name
        res = single_combined_element_exp(model_path, opt.load_sensor_names, opt, explict_sensor=explict_sensor,
                                          alarm_type=alarm_type)
        if res is None:
            return None
        for row in res:
            if row[0] not in all_res:
                all_res[row[0]] = []
            all_res[row[0]].append(row[1:])
    new_avg_result = []
    for alarm, value in all_res.items():
        tmp = np.array(value)[:, 0:5].astype(np.int)
        a, tp, tn, fp, fn = tmp.sum(axis=0)
        acc, tpr, tnr, ref_score, sc = compute_score(tp, fp, tn, fn)
        new_avg_result.append(
            [a, tp, tn, fp, fn, acc, tpr, tnr, ref_score, sc])
    return new_avg_result


def write_csv_file(filename, res):
    with open(filename, 'a+') as f:
        s = ','.join(res)
        f.write(s + '\n')
    return


file_dict = dict()


def init_data_dict(data_path):
    test_datapaths = ["1_fold_test_files_list.txt", "2_fold_test_files_list.txt", "3_fold_test_files_list.txt",
                      "4_fold_test_files_list.txt", "5_fold_test_files_list.txt"]
    for i, file_name in enumerate(test_datapaths):
        with open(data_path + file_name, 'r') as f:
            for line in f:
                file = os.path.join(data_path, line.strip('\n'))
                file_dict[file] = i
    return


all_dataset = defaultdict(WFDB_Loader)


def init_load_data(opt, combined=True):
    if not combined:
        opt.load_sensor_names = ['II', 'V', 'RESP', 'PLETH', 'ABP']
        dataset1 = load_dataset(opt, 'all', False)
        all_dataset['ALL'] = dataset1
        return
    opt.load_sensor_names = ['PLETH', 'RESP', 'ABP']
    dataset1 = load_dataset(opt, 'all', False)
    all_dataset['PRA'] = dataset1

    opt.load_sensor_names = ['II']
    dataset2 = load_dataset(opt, 'all', False)
    all_dataset['II'] = dataset2

    opt.load_sensor_names = ['V']
    dataset3 = load_dataset(opt, 'all', False)
    all_dataset['V'] = dataset3

    opt.load_important_sig = True
    dataset4 = load_dataset(opt, 'all', False)
    opt.load_important_sig = False
    all_dataset['ECG'] = dataset4
    return


def get100_150_epoch_avg_score(opt, alarm_type='all', explict_sensor=False, exp_name="_dgcn_15s", combined=False):
    res = []
    count = 0
    for i in range(100, 151):
        if combined:
            r = run_exp_on_combined_model(opt, alarm_type=alarm_type, exp_name=opt.model_name + exp_name,
                                          model_name=str(i) + '.pth')
        else:
            r = run_cross_val_exp(opt, alarm_type=alarm_type,
                                  explict_sensor=explict_sensor,
                                  exp_name=opt.model_name + exp_name, model_name=str(i) + '.pth')
        if r is None:
            break
        count += 1
        res.append(r[:])
        write_csv_file("./avg_scores/" + opt.model_name +
                       exp_name + ".csv", [str(i)] + [str(v) for v in r])
        print(exp_name, r)
    if count == 0:
        return
    tmp = np.array(res)[:, :5].sum(axis=0) / count
    tmp = [int(v + 0.5) for v in tmp]

    a, tp, fp, tn, fn = tmp[0], tmp[1], tmp[2], tmp[3], tmp[4]
    acc, tpr, tnr, ref_score, sc = compute_score(tp, fp, tn, fn)
    tb = pt.PrettyTable()
    tb.field_names = ['Set', 'Nums', 'TP', 'FP', 'TN',
                      'FN', 'ACC', 'TPR', 'TNR', 'Ref_score', 'Score']
    tb.add_row([opt.model_name + exp_name, a, tp, fp,
               tn, fn, acc, tpr, tnr, ref_score, sc])
    print(tb)
    with open("./avg_scores/" + opt.model_name + exp_name + ".csv", 'a+') as log_file:
        log_file.writelines('\n')
        log_file.write(str(tb) + '\n')
        log_file.writelines('\n')
    return


def exp_cross_val(model_name='deep_modified', prefix='dgcn', combined=False):
    if not combined:
        prefix = '_' + prefix
    # 15s
    opt = gen_options(model_name, use_slice=False,
                      use_norm=False, use_noise=False, use_gnorm=False)
    print(opt.model_name)
    opt.load_sensor_names = ['II', 'V', 'RESP', 'PLETH', 'ABP']
    get100_150_epoch_avg_score(opt, explict_sensor=False, exp_name=prefix + "_15s",
                               combined=combined)

    # 15s 0-1
    opt = gen_options(model_name, use_slice=False,
                      use_norm=True, use_noise=False, use_gnorm=False)
    opt.load_sensor_names = ['II', 'V', 'RESP', 'PLETH', 'ABP']
    get100_150_epoch_avg_score(opt, explict_sensor=False, exp_name=model_name + "_" + prefix + "_15s_norm",
                               combined=combined)

    # 12s slice
    opt = gen_options(model_name, use_slice=True,
                      use_norm=False, use_noise=False, use_gnorm=False)
    opt.load_sensor_names = ['II', 'V', 'RESP', 'PLETH', 'ABP']
    get100_150_epoch_avg_score(opt, explict_sensor=False, exp_name=prefix + "_12s_slice",
                               combined=combined)

    # 12s slice, 0-1
    opt = gen_options(model_name, use_slice=True,
                      use_norm=True, use_noise=False, use_gnorm=False)
    opt.load_sensor_names = ['II', 'V', 'RESP', 'PLETH', 'ABP']
    get100_150_epoch_avg_score(opt, explict_sensor=False, exp_name=prefix + "_12s_slice_norm",
                               combined=combined)

    # 12s slice, nosing
    opt = gen_options(model_name, use_slice=True,
                      use_norm=False, use_noise=True, use_gnorm=False)
    opt.load_sensor_names = ['II', 'V', 'RESP', 'PLETH', 'ABP']
    get100_150_epoch_avg_score(opt, explict_sensor=False, exp_name=prefix + "_12s_slice_nosing",
                               combined=combined)

    # 12s slice, 0-1, nosing
    opt = gen_options(model_name, use_slice=True,
                      use_norm=True, use_noise=True, use_gnorm=False)
    opt.load_sensor_names = ['II', 'V', 'RESP', 'PLETH', 'ABP']
    get100_150_epoch_avg_score(opt, explict_sensor=False, exp_name=prefix + "_12s_slice_norm_nosing",
                               combined=combined)

    # 12s slice, nosing
    opt = gen_options(model_name, use_slice=True,
                      use_norm=False, use_noise=False, use_gnorm=True)
    opt.load_sensor_names = ['II', 'V', 'RESP', 'PLETH', 'ABP']
    get100_150_epoch_avg_score(opt, explict_sensor=False, exp_name=prefix + "_12s_slice_gnorm",
                               combined=combined)
    return


def run_exp_combined_edgcn(opt, model_path, alarm_type='all', fold=-1):
    test_people_preds = defaultdict(list)
    test_people_labels = defaultdict(int)
    test_people_alarm = defaultdict(int)
    for sensors in [['PLETH', 'RESP', 'ABP'], ['II'], ['V'], []]:
        explict_sensor = True if len(sensors) == 1 else False
        if not sensors:
            opt.load_important_sig = True
            opt.load_sensor_names = ['II', 'V']
            ename = 'ECG'
        else:
            opt.load_sensor_names = sensors
            ename = sensors[0] if len(sensors) == 1 else ''.join(
                [c[0] for c in sensors])
            if len(sensors) > 1:
                explict_sensor = False
        # if alarm_type != 'all':
        #     model_path = model_path + '/' + alarm_type
        test_people_preds, test_people_labels, test_people_alarm = single_exp(opt, explict_sensor=explict_sensor,
                                                                              exist_model_path=model_path + ename + '/' + str(
                                                                                  fold + 1) + '/best.pth',
                                                                              alarm_type=alarm_type,
                                                                              test_people_preds=test_people_preds,
                                                                              test_people_labels=test_people_labels,
                                                                              test_people_alarm=test_people_alarm)
        if test_people_labels is None:
            return None
        opt.load_important_sig = False
    test_people_preds = confuse_results(test_people_preds)
    list_preds, list_labels, list_alarm = results_dict_to_list(
        test_people_preds, test_people_labels, test_people_alarm)
    res = score(list_preds, list_labels, list_alarm)
    return res


def run_exp_on_combined_model(opt, alarm_type='all', exp_name='all', model_name='best.pth'):
    res_all = []
    for i in range(5):
        opt.test_files = opt.data_folder + opt.test_datapaths[i]
        res = run_exp_combined_edgcn(opt, '../results/checkpoints/' + exp_name + '/',
                                     alarm_type=alarm_type, fold=i, epoch=model_name)
        if res is None:
            return None
        res_all.append(res)

    res_all = np.array(res_all).swapaxes(0, 1)
    new_avg_result = []
    for i in range(len(res_all)):
        tmp = np.array(res_all[i])[:, 1:6].astype(np.int)
        a, tp, fp, tn, fn = tmp.sum(axis=0)
        acc, tpr, tnr, ref_score, sc = compute_score(tp, fp, tn, fn)
        new_avg_result.append(
            [a, tp, fp, tn, fn, acc, tpr, tnr, ref_score, sc])
    return new_avg_result


def compute_all_exp():
    # exp_cross_val(model_name='dgcn', prefix='dgcn')
    # exp_cross_val(model_name='edgcn', prefix='edgcn')
    exp_cross_val(model_name='edgcn', prefix='', combined=True)
    return


if __name__ == '__main__':
    # 设置Log格式
    compute_all_exp()
    sys.exit(0)
