from sys import exit
import os
from options.test_opt import Opt
from challenge import predict_combined, predict_edgcn, predict_dgcn, predict_combined_valid, all_time, \
    predict_edgcn_valid, predict_dgcn_valid
from sklearn.metrics import confusion_matrix, roc_auc_score
from data_loader.data_read_utils import load_record_use_15s
import numpy as np
from options.OptionFactory import gen_options

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


def compute_score(res_dict):
    for alarm, res in res_dict.items():
        res = np.array(res)
        tn, fp, fn, tp = confusion_matrix(res[:, 2], res[:, 1], labels=[0, 1]).ravel()
        auc = str(round(roc_auc_score(res[:, 2], 1 - res[:, 0]), 4))
        all = tp + tn + fp + fn
        acc = str(round((tn + tp) / all, 4))
        tpr = str(round(tp / max(1, (tp + fn)), 4))
        tnr = str(round(tn / max(1, (tn + fp)), 4))
        ref_score = str(round((tp + fn) / all, 4))
        score = str(round((tp + tn) / (tp + tn + fp + 5 * fn), 4))

        all_score = {"alarm": alarm, "size": all, "tp": tp, "tn": tn, "fp": fp, "fn": fn, "acc": acc, "auc": auc,
                     "tpr": tpr, "tnr": tnr, "ref_score": ref_score, "score": score}
        print(all_score)
    print("")
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


def validtest():
    opt = Opt()
    opt.drop_prob = 0.3
    opt.use_minmax_scale = False
    opt.use_global_minmax = False
    opt.model_name = 'deep_embedded'
    opt.load_sensor_names = ['II', 'V', 'RESP', 'PLETH', 'ABP']
    opt.window_size = 3000
    opt.use_extra = True
    opt.add_noise_prob = 0
    opt.cuda = False

    from data_loader.data_read_utils import load_file, get_record_label
    filelist = load_file("../data/training/RECORDS",
                         "../data/training/")
    init_data_dict("../data/training/")
    all_res = {'ALL': []}
    base_path = './checkpoints/cv_edgcn_all_12s_slice/'
    for record in filelist:
        # if load_record_use_15s(record):
        #     opt = opt_15s_combined()
        #     base_path = './checkpoints/Combined_edgcn_15s/'
        # else:
        #     opt = opt_12s_norm_combined()
        #     base_path = './checkpoints/Combined_edgcn_12s_slice_norm/'

        possi, results = predict_edgcn_valid(opt, record, base_path=base_path,
                                             model_folder=str(file_dict[record] + 1))
        alarm, label = get_record_label(record)
        if alarm not in all_res:
            all_res[alarm] = []
        all_res[alarm].append([possi, results, label])
        all_res['ALL'].append([possi, results, label])
    compute_score(all_res)
    return


def valid_func(opt=gen_options('edgcn', True, False, False, False), base_path='./checkpoints/cv_edgcn_all_12s_slice/',
               predict_func=predict_combined_valid):
    from data_loader.data_read_utils import load_file, get_record_label
    filelist = load_file("../data/training/RECORDS", "../data/training/")
    init_data_dict("../data/training/")
    all_res = {'ALL': []}
    for record in filelist:
        # if load_record_use_15s(record):
        #     opt = opt_15s_combined()
        #     base_path = './checkpoints/Combined_edgcn_15s/'
        # else:
        #     opt = opt_12s_norm_combined()
        #     base_path = './checkpoints/Combined_edgcn_12s_slice_norm/'

        possi, results = predict_func(opt, record, base_path=base_path,
                                      model_folder=str(file_dict[record] + 1))
        alarm, label = get_record_label(record)
        if alarm not in all_res:
            all_res[alarm] = []
        all_res[alarm].append([possi, results, label])
        all_res['ALL'].append([possi, results, label])
    compute_score(all_res)
    return


def valid_combined_model():
    # 15s
    opt = gen_options('edgcn', False, False, False, False)
    base_path = 'checkpoints/Combined_edgcn_15s/'
    valid_func(opt, base_path, predict_combined_valid)

    # 12s slice
    opt = gen_options('edgcn', True, False, False, False)
    base_path = './checkpoints/Combined_edgcn_12s_slice/'
    valid_func(opt, base_path, predict_combined_valid)

    # 12s slice, norm
    opt = gen_options('edgcn', True, True, False, False)
    base_path = './checkpoints/Combined_edgcn_12s_slice_norm/'
    valid_func(opt, base_path, predict_combined_valid)

    # 12s slice,noising
    opt = gen_options('edgcn', True, False, True, False)
    base_path = './checkpoints/Combined_edgcn_12s_slice_noise/'
    valid_func(opt, base_path, predict_combined_valid)

    # 12s slice,norm,noising
    opt = gen_options('edgcn', True, True, True, False)
    base_path = './checkpoints/Combined_edgcn_12s_slice_norm_noise/'
    valid_func(opt, base_path, predict_combined_valid)
    return


def valid_dgcn_model():
    # 15s
    opt = gen_options('dgcn', False, False, False, False)
    base_path = './checkpoints/cv_dgcn_all_15s/'
    valid_func(opt, base_path, predict_dgcn_valid)

    # 12s slice
    opt = gen_options('dgcn', True, False, False, False)
    base_path = './checkpoints/cv_dgcn_all_12s_slice/'
    valid_func(opt, base_path, predict_dgcn_valid)

    # 12s slice, norm
    opt = gen_options('dgcn', True, True, False, False)
    base_path = './checkpoints/cv_dgcn_all_12s_slice_norm/'
    valid_func(opt, base_path, predict_dgcn_valid)

    # 12s slice,noising
    opt = gen_options('dgcn', True, False, True, False)
    base_path = './checkpoints/cv_dgcn_all_12s_slice_nosing/'
    valid_func(opt, base_path, predict_dgcn_valid)

    # 12s slice,norm,noising
    opt = gen_options('dgcn', True, True, True, False)
    base_path = './checkpoints/cv_dgcn_all_12s_slice_norm_nosing/'
    valid_func(opt, base_path, predict_dgcn_valid)
    return


def valid_edgcn_model():
    # 15s
    # {'alarm': 'ALL', 'size': 750, 'tp': 272, 'tn': 367, 'fp': 89, 'fn': 22, 'acc': '0.852', 'auc': '0.9072',
    #  'tpr': '0.9252', 'tnr': '0.8048', 'ref_score': '0.392', 'score': '0.7625'}
    # {'alarm': 'VTA', 'size': 341, 'tp': 77, 'tn': 211, 'fp': 41, 'fn': 12, 'acc': '0.8446', 'auc': '0.876',
    #  'tpr': '0.8652', 'tnr': '0.8373', 'ref_score': '0.261', 'score': '0.7404'}
    # {'alarm': 'ASY', 'size': 122, 'tp': 18, 'tn': 79, 'fp': 21, 'fn': 4, 'acc': '0.7951', 'auc': '0.8782',
    #  'tpr': '0.8182', 'tnr': '0.79', 'ref_score': '0.1803', 'score': '0.7029'}
    # {'alarm': 'ETC', 'size': 140, 'tp': 129, 'tn': 7, 'fp': 2, 'fn': 2, 'acc': '0.9714', 'auc': '0.9873',
    #  'tpr': '0.9847', 'tnr': '0.7778', 'ref_score': '0.9357', 'score': '0.9189'}
    # {'alarm': 'VFB', 'size': 58, 'tp': 4, 'tn': 38, 'fp': 14, 'fn': 2, 'acc': '0.7241', 'auc': '0.6875',
    #  'tpr': '0.6667', 'tnr': '0.7308', 'ref_score': '0.1034', 'score': '0.6364'}
    # {'alarm': 'EBR', 'size': 89, 'tp': 44, 'tn': 32, 'fp': 11, 'fn': 2, 'acc': '0.8539', 'auc': '0.8572',
    #  'tpr': '0.9565', 'tnr': '0.7442', 'ref_score': '0.5169', 'score': '0.7835'}
    # opt = gen_options('edgcn', False, False, False, False)
    # base_path = './checkpoints/cv_edgcn_all_15s/'
    # valid_func(opt, base_path, predict_edgcn_valid)

    # # 12s slice
    # opt = gen_options('edgcn', True, False, False, False)
    # base_path = './checkpoints/deep_embedded_edgcn_12s_slice/'
    # valid_func(opt, base_path, predict_edgcn_valid)

    # # 12s slice, norm
    # opt = gen_options('edgcn', True, True, False, False)
    # base_path = './checkpoints/cv_edgcn_all_12s_slice_norm/'
    # valid_func(opt, base_path, predict_edgcn_valid)

    # # 12s slice,noising
    # opt = gen_options('edgcn', True, False, True, False)
    # base_path = './checkpoints/cv_edgcn_all_12s_slice_noising/'
    # valid_func(opt, base_path, predict_edgcn_valid)

    # 12s slice,norm,noising
    opt = gen_options('edgcn', True, True, True, False)
    base_path = './checkpoints/cv_edgcn_all_12s_slice_norm_noising/'
    valid_func(opt, base_path, predict_edgcn_valid)
    return


if __name__ == "__main__":
    # valid_dgcn_model()
    valid_edgcn_model()
    # valid_combined_model()
    exit(0)
