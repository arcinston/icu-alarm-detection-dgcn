from sys import exit
import os
from options.test_opt import Opt
from challenge import predict_combined, predict_edgcn, predict_dgcn, predict_combined_valid, all_time, \
    predict_edgcn_valid, predict_dgcn_valid
from sklearn.metrics import confusion_matrix, roc_auc_score
from data_loader.data_read_utils import load_record_use_15s
import numpy as np

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


def ptest():
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
    from data_loader.data_read_utils import load_file, get_record_label
    filelist = load_file("/Users/yuq/PycharmProjects/ecg_marker/data/training/RECORDS",
                         "/Users/yuq/PycharmProjects/ecg_marker/data/training/")
    tp, tn, fp, fn = 0, 0, 0, 0
    with open("answers.txt", "w+", encoding="utf-8") as f:
        for record in filelist:
            if load_record_use_15s(record):
                opt = opt_15s_combined()
                base_path = 'checkpoints/Combined_edgcn_15s_bak/'
            else:
                opt = opt_12s_norm_combined()
                base_path = './checkpoints/Combined_edgcn_12s_slice_norm/'

            output_file = os.path.basename(record)
            p, results = predict_combined(opt, record, base_path=base_path)
            f.write(output_file + ',' + str(results) + '\n')
            alarm, label = get_record_label(record)
            if results == label:
                if results == 0:
                    tn += 1
                else:
                    tp += 1
            else:
                if results == 0:
                    fn += 1
                else:
                    fp += 1
    score = (tp + tn) / (tp + tn + fp + 5 * fn)
    print(score)
    return


def compute_score(res_dict):
    for alarm, res in res_dict.items():
        res = np.array(res)
        tn, fp, fn, tp = confusion_matrix(res[:, 2], res[:, 1], labels=[0, 1]).ravel()
        auc = round(roc_auc_score(res[:, 2], 1 - res[:, 0]), 4)
        all = tp + tn + fp + fn
        acc = round((tn + tp) / all, 4)
        tpr = round(tp / max(1, (tp + fn)), 4)
        tnr = round(tn / max(1, (tn + fp)), 4)
        ref_score = round((tp + fn) / all, 4)
        score = round((tp + tn) / (tp + tn + fp + 5 * fn), 4)

        all_score = {"alarm": alarm, "size": all, "tp": tp, "tn": tn, "fp": fp, "fn": fn, "acc": acc, "auc": auc,
                     "tpr": tpr, "tnr": tnr, "ref_score": ref_score, "score": score}
        print(all_score)
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


if __name__ == "__main__":
    validtest()
    # ptest()
    exit(0)
