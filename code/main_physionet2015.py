import logging
import sys

import torch.nn.functional as F
import torch as th
from torch import optim
from torch_ext.util import set_random_seeds

from experiments.experiment_physionet import Experiment
from experiments.stopcriteria import MaxEpochs, Or, NoIncrease
from torch_ext.constraints import MaxNormDefaultConstraint
from experiments.monitors import LossMonitor, ScoreMonitor

from options.default_opt import Opt
from datasets.wfdb_dataset import WFDB_Loader

from models.model_loader import load_model
from experiments_scores import run_exp_combined_edgcn, single_combined_element_exp
from compute_socore import log_cv_results
from options.OptionFactory import gen_options

log = logging.getLogger(__name__)
log.setLevel('DEBUG')


def compute_score(tp, fp, tn, fn):
    acc = (tn + tp) / (tp + tn + fp + fn)
    tpr = tp / max(1, (tp + fn))
    tnr = tn / max(1, (tn + fp))
    ref_score = (tp + fn) / (tp + tn + fp + fn)
    score = (tp + tn) / (tp + tn + fp + 5 * fn)
    print("ACC:", acc)
    print("TPR:", tpr)
    print("tnr:", tnr)
    print("score:", score)
    return acc, tpr, tnr, ref_score, score


def load_dataset(opt, alarm_type='all', explict_sensor=True):
    log.info("Loading dataset...")
    dataset = WFDB_Loader(opt)
    dataset.filter_files(alarm_type, explict_sensor)
    return dataset


def run_exp_on_wfdb_dataset(opt, alarm_type='all', explict_sensor=True, exp_name='all'):
    dataset = load_dataset(opt, alarm_type, explict_sensor)
    set_random_seeds(opt.np_to_seed, cuda=opt.cuda)
    if opt.debug:
        opt.max_epochs = 4
    exp = single_exp(dataset, opt, exp_name)
    return


def run_cross_val_exp(opt, alarm_type='all', explict_sensor=True, exp_name='all'):
    train_datapaths = ["1_fold_train_files_list.txt", "2_fold_train_files_list.txt", "3_fold_train_files_list.txt",
                       "4_fold_train_files_list.txt", "5_fold_train_files_list.txt"]
    test_datapaths = ["1_fold_test_files_list.txt", "2_fold_test_files_list.txt", "3_fold_test_files_list.txt",
                      "4_fold_test_files_list.txt", "5_fold_test_files_list.txt"]
    res_dict = {}
    for i in range(5):
        log.info("value of i is %d", i)
        opt.train_files = opt.data_folder + train_datapaths[i]
        opt.test_files = opt.data_folder + test_datapaths[i]
        ename = exp_name + "/" + str(i + 1)
        log.info("exp name is %s", ename)
        run_exp_on_wfdb_dataset(opt, alarm_type, explict_sensor, ename)
        base_path = '../results/checkpoints/' + ename
        print(base_path)
        res = single_combined_element_exp(base_path + "/best.pth", opt.load_sensor_names, opt,
                                          explict_sensor=explict_sensor, alarm_type=alarm_type)
        if res is None:
            return None
        for r in range(len(res)):
            alarm = res[r][0]
            if alarm not in res_dict:
                res_dict[alarm] = []
            res_dict[alarm].append([i + 1] + res[r])
    tname = exp_name.replace('/', '_')
    log_cv_results(res_dict, tname)
    return


def run_exp_all(opt):
    opt.load_sensor_names = ['II', 'V']
    run_exp_on_wfdb_dataset(opt, alarm_type='all', explict_sensor=True)
    for alarm in opt.all_alarm_type:
        run_exp_on_wfdb_dataset(opt, alarm_type=alarm, explict_sensor=True)
    opt.load_sensor_names = ['I', 'II', 'III', 'V',
                             'aVL', 'aVR', 'aVF', 'RESP', 'PLETH', 'MCL', 'ABP']
    run_exp_on_wfdb_dataset(opt, alarm_type='all', explict_sensor=False)
    for alarm in opt.all_alarm_type:
        run_exp_on_wfdb_dataset(opt, alarm_type=alarm, explict_sensor=False)
    return


def single_exp(dataset, opt, exp_name):
    opt.n_classes = 2
    opt.input_nc = len(
        opt.load_sensor_names) if not opt.load_important_sig else 1
    opt.input_length = opt.window_size
    if opt.use_extra:
        opt.extra_length = dataset.get_extro_len()
        print("extro info lens", opt.extra_length)

    model = load_model(opt)
    # 定义优化器
    optimizer = optim.Adam(model.parameters(), weight_decay=opt.weight_decay,
                           lr=opt.lr)
    monitors = [LossMonitor(), ScoreMonitor(), ]
    model_constraint = MaxNormDefaultConstraint()

    if opt.use_extra:
        def loss_function(preds, targets): return F.nll_loss(th.squeeze(preds, dim=1),
                                                             targets)
    else:
        def loss_function(preds, targets): return F.nll_loss(th.squeeze(preds, dim=2),
                                                             targets)
    do_early_stop = True
    stop_criterion = Or([MaxEpochs(opt.max_epoch),
                         NoIncrease('ALL_train_ACC', opt.max_increase_epoch)])
    remember_best_column = 'ALL_test_Score'
    path = '../results/checkpoints/' + exp_name
    import os
    if not os.path.exists(path):
        os.makedirs(path)
    import os
    if not os.path.exists(path):
        os.makedirs(path)
    exp = Experiment(opt, model, datasets=dataset,
                     loss_function=loss_function, optimizer=optimizer,
                     model_constraint=model_constraint,
                     monitors=monitors,
                     stop_criterion=stop_criterion,
                     remember_best_column=remember_best_column,
                     do_early_stop=do_early_stop,
                     training_increase=True,
                     save_path=path)
    exp.run()
    return exp


def split_alarm_exp():
    opt = Opt()
    run_exp_on_wfdb_dataset(opt, alarm_type='all', explict_sensor=False)
    for alarm in opt.all_alarm_type:
        run_exp_on_wfdb_dataset(opt, alarm_type=alarm, explict_sensor=True)
    return


def exp_single_model(model_name='dgcn', alarm_type='all'):
    # 15s
    opt = gen_options(model_name, False, False, False, False)
    opt.load_sensor_names = ['II', 'V', 'RESP', 'PLETH', 'ABP']
    run_exp_on_wfdb_dataset(opt, alarm_type=alarm_type, explict_sensor=False,
                            exp_name="split_dataset_" + model_name + '_' + alarm_type + "_15s")

    # 15s 0-1
    opt = gen_options(model_name, False, True, False, False)
    opt.load_sensor_names = ['II', 'V', 'RESP', 'PLETH', 'ABP']
    run_exp_on_wfdb_dataset(opt, alarm_type=alarm_type, explict_sensor=False,
                            exp_name="split_dataset_" + model_name + '_' + alarm_type + "_15s_norm")

    # 12s slice
    opt = gen_options(model_name, True, False, False, False)
    opt.load_sensor_names = ['II', 'V', 'RESP', 'PLETH', 'ABP']
    run_exp_on_wfdb_dataset(opt, alarm_type=alarm_type, explict_sensor=False,
                            exp_name="split_dataset_" + model_name + '_' + alarm_type + "_12s_slice")

    # 12s slice, 0-1
    opt = gen_options(model_name, True, True, False, False)
    opt.load_sensor_names = ['II', 'V', 'RESP', 'PLETH', 'ABP']
    run_exp_on_wfdb_dataset(opt, alarm_type=alarm_type, explict_sensor=False,
                            exp_name="split_dataset_" + model_name + '_' + alarm_type + "_12s_slice_norm")

    # 12s slice, nosing
    opt = gen_options(model_name, True, False, True, False)
    opt.load_sensor_names = ['II', 'V', 'RESP', 'PLETH', 'ABP']
    run_exp_on_wfdb_dataset(opt, alarm_type=alarm_type, explict_sensor=False,
                            exp_name="split_dataset_" + model_name + '_' + alarm_type + "_12s_slice_noise")

    # 12s slice, 0-1, nosing
    opt = gen_options(model_name, True, True, True, False)
    opt.load_sensor_names = ['II', 'V', 'RESP', 'PLETH', 'ABP']
    run_exp_on_wfdb_dataset(opt, alarm_type=alarm_type, explict_sensor=False,
                            exp_name="split_dataset_" + model_name + '_' + alarm_type + "_12s_slice_norm_noise")
    return


def exp_single_model_cross_val(model_name='dgcn', alarm_type='all'):
    # 15s
    # opt = gen_options(model_name, False, False, False, False)
    # opt.load_sensor_names = ['II', 'V', 'RESP', 'PLETH', 'ABP']
    # run_cross_val_exp(opt, alarm_type=alarm_type, explict_sensor=False,
    #                   exp_name="cv_" + model_name + '_' + alarm_type + "_15s")

    # # 15s 0-1
    # opt = gen_options(model_name, False, True, False, False)
    # opt.load_sensor_names = ['II', 'V', 'RESP', 'PLETH', 'ABP']
    # run_cross_val_exp(opt, alarm_type=alarm_type, explict_sensor=False,
    #                   exp_name="cv_" + model_name + '_' + alarm_type + "_15s_norm")
    #
    # # 12s slice
    # opt = gen_options(model_name, True, False, False, False)
    # opt.load_sensor_names = ['II', 'V', 'RESP', 'PLETH', 'ABP']
    # run_cross_val_exp(opt, alarm_type=alarm_type, explict_sensor=False,
    #                   exp_name="cv_" + model_name + '_' + alarm_type + "_12s_slice")
    #
    # # 12s slice, 0-1
    # opt = gen_options(model_name, True, True, False, False)
    # opt.load_sensor_names = ['II', 'V', 'RESP', 'PLETH', 'ABP']
    # run_cross_val_exp(opt, alarm_type=alarm_type, explict_sensor=False,
    #                   exp_name="cv_" + model_name + '_' + alarm_type + "_12s_slice_norm")
    #
    # # 12s slice, nosing
    # opt = gen_options(model_name, True, False, True, False)
    # opt.load_sensor_names = ['II', 'V', 'RESP', 'PLETH', 'ABP']
    # run_cross_val_exp(opt, alarm_type=alarm_type, explict_sensor=False,
    #                   exp_name="cv_" + model_name + '_' + alarm_type + "_12s_slice_nosing")

    # 12s slice, 0-1, nosing
    opt = gen_options(model_name, True, True, True, False)
    opt.load_sensor_names = ['II', 'V', 'RESP', 'PLETH', 'ABP']
    run_cross_val_exp(opt, alarm_type=alarm_type, explict_sensor=False,
                      exp_name="cv_" + model_name + '_' + alarm_type + "_12s_slice_norm_nosing")
    return


def tmp_dgcn():
    opt = gen_options("dgcn", True, True, True, False)
    opt.load_sensor_names = ['II', 'V', 'RESP', 'PLETH', 'ABP']
    run_cross_val_exp(opt, alarm_type="all", explict_sensor=False,
                      exp_name="cv_dgcn_all_12s_slice_norm_nosing")
    return


def tmp_edgcn():

    opt = gen_options("edgcn", True, True, True, False)
    opt.load_sensor_names = ['II', 'V', 'RESP', 'PLETH', 'ABP']
    run_cross_val_exp(opt, alarm_type="all", explict_sensor=False,
                      exp_name="cv_edgcn_all_12s_slice_norm_noising")
    return


def exp_combined():
    for alarm in ['all']:  # You can choose different alarm
        opt = Opt()
        opt.drop_prob = 0.3
        opt.load_sensor_names = ['PLETH', 'RESP', 'ABP']
        opt.use_minmax_scale = True
        opt.model_name = 'edgcn'  # 'edgcn'
        opt.use_extra = True
        opt.add_noise_prob = 0
        opt.window_size = 3750
        run_exp_on_wfdb_dataset(opt, alarm_type=alarm, explict_sensor=False)

        opt = Opt()
        opt.drop_prob = 0.3
        opt.load_sensor_names = ['II']
        opt.use_minmax_scale = True
        opt.model_name = 'edgcn'  # 'edgcn'
        opt.use_extra = True
        opt.add_noise_prob = 0
        opt.window_size = 3750
        run_exp_on_wfdb_dataset(opt, alarm_type=alarm, explict_sensor=True)

        opt = Opt()
        opt.drop_prob = 0.3
        opt.load_sensor_names = ['V']
        opt.use_minmax_scale = True
        opt.model_name = 'edgcn'  # 'edgcn'
        opt.use_extra = True
        opt.add_noise_prob = 0
        opt.window_size = 3750
        run_exp_on_wfdb_dataset(opt, alarm_type=alarm, explict_sensor=True)

        opt = Opt()
        opt.drop_prob = 0.3
        opt.load_important_sig = True
        opt.use_minmax_scale = True
        opt.model_name = 'edgcn'  # 'edgcn'这里需要再训练
        opt.use_extra = True
        opt.add_noise_prob = 0
        opt.window_size = 3750
        run_exp_on_wfdb_dataset(opt, alarm_type=alarm, explict_sensor=False)

    return


def run_exp_on_combined_model(opt, model='edgcn', exp_name='all'):
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

        run_cross_val_exp(opt, alarm_type='all', explict_sensor=explict_sensor,
                          exp_name="Combined_" + model + '_' + exp_name + '/' + ename)
        opt.load_important_sig = False
    res_dict = {}
    for i in range(5):
        opt.test_files = opt.data_folder + opt.test_datapaths[i]
        res = run_exp_combined_edgcn(
            opt, '../results/checkpoints/' + "Combined_" + model + '_' + exp_name + '/', 'all', i)
        for r in range(len(res)):
            alarm = res[r][0]
            if alarm not in res_dict:
                res_dict[alarm] = []
            res_dict[alarm].append([i + 1] + res[r])
    log_cv_results(res_dict, exp_name)
    return


def exp_combined_model_cross_val(model='edgcn'):
    # opt = gen_options(model, False, False, False, False)
    # run_exp_on_combined_model(opt, model, "15s")

    # opt = gen_options(model, False, True, False, False)
    # run_exp_on_combined_model(opt, model, "15s_norm")

    # opt = gen_options(model, True, False, False, False)
    # run_exp_on_combined_model(opt, model, "12s_slice")

    # opt = gen_options(model, True, True, False, False)
    # run_exp_on_combined_model(opt, model, "12s_slice_norm")
    
    # opt = gen_options(model, True, False, True, False)
    # run_exp_on_combined_model(opt, model, "12s_slice_nosing")

    opt = gen_options(model, True, True, True, False)
    run_exp_on_combined_model(opt, model, "12s_slice_norm_nosing")
    return


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(levelname)s : %(message)s',
                        level=logging.DEBUG, stream=sys.stdout)

    all_alarm_type = ['Ventricular_Tachycardia', 'Tachycardia', 'Ventricular_Flutter_Fib', 'Bradycardia',
                      'Asystole']

    # exp_single_model_cross_val("dgcn")
    # exp_single_model_cross_val("edgcn")
    # exp_combined_model_cross_val("edgcn")
    # tmp_dgcn()
    tmp_edgcn()

    sys.exit(0)
