from sys import exit
import prettytable as pt
import time
import numpy as np


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
    return round(acc, 4), round(tpr, 4), round(tnr, 4), round(ref_score, 4), round(score, 4)


def log_cv_results(res_dict, exp_name):
    table = pt.PrettyTable()
    table.field_names = ['Set', 'Alarm', 'Nums', 'TP', 'TN', 'FP', 'FN', 'ACC', 'TPR', 'TNR', 'Ref_score',
                         'Score']
    with open('../results/'+exp_name.replace('/', '_') + '.txt', 'a+') as log_file:
        time_str = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        for alarm, res in res_dict.items():
            stb = pt.PrettyTable()
            stb.field_names = ['Fold', 'Alarm', 'Nums', 'TP', 'TN', 'FP', 'FN', 'ACC', 'AUC', 'TPR', 'TNR', 'Ref_score',
                               'Score']
            for r in range(len(res)):
                stb.add_row(res[r])
            print(stb)
            log_file.write(str(stb) + '\n')

            tmp = np.array(res)[:, 2:7].astype(np.int)
            a, tp, tn, fp, fn = tmp.sum(axis=0)
            acc, tpr, tnr, ref_score, sc = compute_score(tp, fp, tn, fn)
            table.add_row([exp_name, alarm, a, tp, tn, fp, fn, acc, tpr, tnr, ref_score, sc])

        print(table)
        log_file.write(time_str + ': all result \n')
        log_file.write(str(table) + '\n')
        log_file.writelines('\n')
    return


if __name__ == "__main__":
    compute_score(280, 89, 367, 14)
    exit(0)
