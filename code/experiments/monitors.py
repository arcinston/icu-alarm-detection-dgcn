import numpy as np
import time
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.metrics import confusion_matrix
import logging

# logging.basicConfig(filename='./LOG/main.log')
alarm_map = {0: 'VTA', 1: 'ETC', 2: 'VFB', 3: 'EBR', 4: 'ASY'}
log = logging.getLogger(__name__)


class MisclassMonitor(object):
    """
    Monitor the examplewise misclassification rate.

    Parameters
    ----------
    col_suffix: str, optional
        Name of the column in the monitoring output.
    threshold_for_binary_case: bool, optional
        In case of binary classification with only one output prediction
        per target, define the threshold for separating the classes, i.e.
        0.5 for sigmoid outputs, or np.log(0.5) for log sigmoid outputs
    """

    def __init__(self, col_suffix="misclass", threshold_for_binary_case=0.5):
        self.col_suffix = col_suffix
        self.threshold_for_binary_case = threshold_for_binary_case

    def monitor_epoch(self, ):
        return

    def monitor_set(
            self,
            setname,
            all_preds,
            all_losses,
            all_batch_sizes,
            all_targets,
            all_targets_alarm_type
    ):
        all_pred_labels = []
        all_target_labels = []
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        for preds, targets in zip(all_preds, all_targets):
            if preds.ndim == 2:
                pred_labels = np.int32(preds[0, 0] < self.threshold_for_binary_case)
            else:
                pred_labels = np.int32(preds[0] < self.threshold_for_binary_case)
            targets_labels = targets
            all_pred_labels.append(pred_labels)
            all_target_labels.append(targets_labels)

        all_pred_labels = np.array(all_pred_labels)
        all_target_labels = np.array(all_target_labels)
        assert all_pred_labels.shape == all_target_labels.shape

        misclass = 1 - np.mean(all_target_labels == all_pred_labels)
        column_name = "{:s}_{:s}".format(setname, self.col_suffix)
        return {column_name: float(misclass)}


def compute_score(tp, fp, tn, fn):
    all = tp + tn + fp + fn
    acc = (tn + tp) / all
    tpr = tp / max(1, (tp + fn))
    tnr = tn / max(1, (tn + fp))
    ref_score = (tp + fn) / all
    score = (tp + tn) / (tp + tn + fp + 5 * fn)
    return acc, tpr, tnr, ref_score, score


def get_alarm_samples(preds, targets, alarm_types):
    alarm_preds = {'ALL': []}
    alarm_targets = {'ALL': []}
    for i, alarm in enumerate(alarm_types):
        pos = np.argmax(alarm)
        if alarm_map[pos] not in alarm_preds:
            alarm_preds[alarm_map[pos]] = []
            alarm_targets[alarm_map[pos]] = []
        alarm_preds[alarm_map[pos]].append(preds[i])
        alarm_targets[alarm_map[pos]].append(targets[i])
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
    alarm_preds = {'VTA': [], 'ETC': [],
                   'VFB': [], 'EBR': [],
                   'ASY': []}
    alarm_targets = {'VTA': [], 'ETC': [],
                     'VFB': [], 'EBR': [],
                     'ASY': []}

    for i, alarm in enumerate(alarm_types):
        pos = np.argmax(alarm)
        alarm_preds[alarm_map[pos]].append(preds[i])
        alarm_targets[alarm_map[pos]].append(targets[i])

    for alarm, preds in alarm_preds.items():
        info = confusion_matrix(targets, preds, labels=[0, 1]).ravel()
        alarm_cm[alarm] += info
    return alarm_cm


class ScoreMonitor(object):
    def __init__(self, col_suffix="Score", threshold_for_binary_case=0.5):
        self.col_suffix = col_suffix
        self.threshold_for_binary_case = threshold_for_binary_case

    def monitor_epoch(self, ):
        return

    def monitor_set(
            self,
            setname,
            all_preds,
            all_losses,
            all_batch_sizes,
            all_targets,
            all_targets_alarm_type
    ):
        preds = np.concatenate(all_preds, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        alarm_types = np.concatenate(all_targets_alarm_type, axis=0)
        alarm_preds, alarm_targets = get_alarm_samples(preds, targets, alarm_types)

        all_scores = {}
        for alarm, preds in alarm_preds.items():
            preds = np.array(preds)
            if preds.ndim == 3:
                p = preds.squeeze(axis=2)[:, 1]
            else:
                p = preds[:, 1]

            t = alarm_targets[alarm]
            # if np.any(np.isnan(p)):
            #     raise Exception("preds have nan value!!!")
            try:
                auc = roc_auc_score(t, p)
            except ValueError:
                auc = 0

            p = np.int32(p > self.threshold_for_binary_case)
            tn, fp, fn, tp = confusion_matrix(t, p, labels=[0, 1]).ravel()
            acc, tpr, tnr, ref_score, score = compute_score(tp, fp, tn, fn)
            acc_name = "{:s}_{:s}_{:s}".format(alarm, setname, 'ACC')
            auc_name = "{:s}_{:s}_{:s}".format(alarm, setname, 'AUC')
            score_name = "{:s}_{:s}_{:s}".format(alarm, setname, 'Score')
            tpr_name = "{:s}_{:s}_{:s}".format(alarm, setname, 'TPR')
            tnr_name = "{:s}_{:s}_{:s}".format(alarm, setname, 'TNR')
            tp_name = "{:s}_{:s}_{:s}".format(alarm, setname, 'TP')
            tn_name = "{:s}_{:s}_{:s}".format(alarm, setname, 'TN')
            fp_name = "{:s}_{:s}_{:s}".format(alarm, setname, 'FP')
            fn_name = "{:s}_{:s}_{:s}".format(alarm, setname, 'FN')
            total_name = "{:s}_{:s}_{:s}".format(alarm, setname, 'Total')
            all_scores[total_name] = tn + fp + fn + tp
            all_scores[tp_name] = tp
            all_scores[tn_name] = tn
            all_scores[fp_name] = fp
            all_scores[fn_name] = fn
            all_scores[acc_name] = acc
            all_scores[auc_name] = auc
            all_scores[tpr_name] = tpr
            all_scores[tnr_name] = tnr
            all_scores[score_name] = score

        return all_scores


def perf_measure(y_true, y_pred):
    TP, FP, TN, FN = 0, 0, 0, 0

    for i in range(len(y_true)):
        if y_true[i] == 1 and y_pred[i] == 1:
            TP += 1
        if y_true[i] == 0 and y_pred[i] == 1:
            FP += 1
        if y_true[i] == 0 and y_pred[i] == 0:
            TN += 1
        if y_true[i] == 1 and y_pred[i] == 0:
            FN += 1

    return TP, FP, TN, FN


class LossMonitor(object):
    """
    Monitor the examplewise loss.
    """

    def monitor_epoch(self, ):
        return

    def monitor_set(
            self,
            setname,
            all_preds,
            all_losses,
            all_batch_sizes,
            all_targets,
            all_targets_alarm_type
    ):
        batch_weights = np.array(all_batch_sizes) / float(
            np.sum(all_batch_sizes)
        )
        loss_per_batch = [np.mean(loss) for loss in all_losses]
        mean_loss = np.sum(batch_weights * loss_per_batch)
        column_name = "ALL_{:s}_loss".format(setname)
        return {column_name: mean_loss}


class RuntimeMonitor(object):
    """
    Monitor the runtime of each epoch.

    First epoch will have runtime 0.
    """

    def __init__(self):
        self.last_call_time = None

    def monitor_epoch(self, ):
        cur_time = time.time()
        if self.last_call_time is None:
            # just in case of first call
            self.last_call_time = cur_time
        epoch_runtime = cur_time - self.last_call_time
        self.last_call_time = cur_time
        return {"runtime": epoch_runtime}

    def monitor_set(
            self,
            setname,
            all_preds,
            all_losses,
            all_batch_sizes,
            all_targets,
            all_targets_alarm_type
    ):
        return {}
