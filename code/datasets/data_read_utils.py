import os
import sys
from sys import exit
import logging
from sklearn import preprocessing
import numpy as np
import shutil
import wfdb
from sklearn.model_selection import KFold
import re
from collections import defaultdict
import copy

log = logging.getLogger(__name__)
log.setLevel('DEBUG')

r'''
本文件仅负责数据的读取
'''

all_sensor_name = ['I', 'II', 'III', 'V', 'aVL', 'aVR', 'aVF', 'RESP', 'PLETH', 'MCL', 'ABP']
important_sig = ['I', 'II', 'III', 'V', 'aVL', 'aVR', 'aVF', 'MCL']
other_sig = ['RESP', 'PLETH', 'ABP']
all_alarm_id = {'Ventricular_Tachycardia': [1, 0, 0, 0, 0], 'Tachycardia': [0, 1, 0, 0, 0],
                'Ventricular_Flutter_Fib': [0, 0, 1, 0, 0], 'Bradycardia': [0, 0, 0, 1, 0],
                'Asystole': [0, 0, 0, 0, 1]}
all_alarm_num = {'Ventricular_Tachycardia': 0, 'Tachycardia': 0,
                 'Ventricular_Flutter_Fib': 0, 'Bradycardia': 0,
                 'Asystole': 0}


def split_dataset_bac(origin_path, train_path, test_path, test_size=0.2, seed=20200105):
    fileList = os.listdir(origin_path)
    fileList = sorted(f for f in fileList if os.path.isfile(os.path.join(origin_path, f)) and f.endswith(".mat"))
    files = []
    for s in fileList:
        files.append(s[:-4])
    np.random.seed(seed)
    np.random.shuffle(files)
    index = int(test_size * len(files))
    for f in files[:index]:
        shutil.copyfile(os.path.join(origin_path, f) + '.mat', os.path.join(test_path, f) + '.mat')
        shutil.copyfile(os.path.join(origin_path, f) + '.hea', os.path.join(test_path, f) + '.hea')
    for f in files[index:]:
        shutil.copyfile(os.path.join(origin_path, f) + '.mat', os.path.join(train_path, f) + '.mat')
        shutil.copyfile(os.path.join(origin_path, f) + '.hea', os.path.join(train_path, f) + '.hea')

    return


def split_dataset(origin_path, test_size=0.2, seed=202005):
    fileList = os.listdir(origin_path)
    fileList = sorted(f for f in fileList if os.path.isfile(os.path.join(origin_path, f)) and f.endswith(".mat"))
    files = []
    for s in fileList:
        files.append(s[:-4])
    np.random.seed(seed)
    np.random.shuffle(files)
    index = int(test_size * len(files))
    with open(os.path.join(origin_path, 'test_files_list.txt'), 'w+') as test_file:
        for f in files[:index]:
            test_file.writelines(f + '\n')
    with open(os.path.join(origin_path, 'train_files_list.txt'), 'w+') as train_file:
        for f in files[index:]:
            train_file.writelines(f + '\n')
    return


def split_dataset_balance(origin_path, test_size=0.2, seed=2020):
    fileList = os.listdir(origin_path)
    fileList = sorted(f for f in fileList if os.path.isfile(os.path.join(origin_path, f)) and f.endswith(".mat"))
    filedict = dict()
    test_files = []
    train_files = []
    for s in fileList:
        p = os.path.join(origin_path, s[:-4])
        record = load_record(p)
        comments = record.comments
        if comments[0] not in filedict:
            filedict[comments[0]] = [[], []]
        if record.comments[1] == 'True alarm':
            filedict[comments[0]][0].append(s[:-4])
        else:
            filedict[comments[0]][1].append(s[:-4])

    np.random.seed(seed)
    for k, v in filedict.items():
        index_positive = int(test_size * len(v[0]))
        index_negative = int(test_size * len(v[1]))
        np.random.shuffle(v[0])
        np.random.shuffle(v[1])
        test_files = test_files + v[0][:index_positive] + v[1][:index_negative]
        train_files = train_files + v[0][index_positive:] + v[1][index_negative:]
    np.random.shuffle(test_files)
    np.random.shuffle(train_files)
    with open(os.path.join(origin_path, 'test_files_list.txt'), 'w+') as test_file:
        for f in test_files:
            test_file.writelines(f + '\n')

    with open(os.path.join(origin_path, 'train_files_list.txt'), 'w+') as train_file:
        for f in train_files:
            train_file.writelines(f + '\n')
    return


def stat_data(origin_path):
    fileList = os.listdir(origin_path)
    fileList = sorted(f for f in fileList if os.path.isfile(os.path.join(origin_path, f)) and f.endswith(".mat"))
    positive = 0
    negative = 0
    for s in fileList:
        p = os.path.join(origin_path, s[:-4])
        record = load_record(p)
        if record.comments[1] == 'False alarm':
            positive += 1
        else:
            negative += 1

    return positive, negative, positive / negative


def stat_data_use_file(filename, data_path):
    fileList = []
    with open(filename, 'r') as f:
        for line in f:
            fileList.append(os.path.join(data_path, line.strip('\n')))
    positive = 0
    negative = 0
    for s in fileList:
        record = load_record(s)
        if record.comments[1] == 'False alarm':
            positive += 1
        else:
            negative += 1

    return positive, negative, positive / negative


def cross_val_dataset(origin_path, n_splits=5, seed=20200105):
    fileList = os.listdir(origin_path)
    pattern = re.compile(r'\d+_fold_\w+')
    for f in fileList:
        if f.endswith('.txt'):
            if pattern.match(f) is not None:
                os.remove(os.path.join(origin_path, f))

    fileList = sorted(f for f in fileList if os.path.isfile(os.path.join(origin_path, f)) and f.endswith(".mat"))
    files = []
    for s in fileList:
        files.append(s[:-4])
    np.random.seed(seed)
    np.random.shuffle(files)
    kf = KFold(n_splits=n_splits)
    i = 1
    for train_index, test_index in kf.split(files):
        files_train, files_test = np.array(files)[train_index], np.array(files)[test_index]

        with open(os.path.join(origin_path, str(i) + '_fold_test_files_list.txt'), 'w+') as test_file:
            for f in files_test:
                test_file.writelines(f + '\n')

        with open(os.path.join(origin_path, str(i) + '_fold_train_files_list.txt'), 'w+') as train_file:
            for f in files_train:
                train_file.writelines(f + '\n')
        i += 1

    return


def cross_val_dataset_balance(origin_path, n_splits=5, seed=2020):
    fileList = os.listdir(origin_path)

    fileList = sorted(f for f in fileList if os.path.isfile(os.path.join(origin_path, f)) and f.endswith(".mat"))
    filedict = dict()
    for s in fileList:
        p = os.path.join(origin_path, s[:-4])
        record = load_record(p)
        comments = record.comments
        if comments[0] not in filedict:
            filedict[comments[0]] = [[], []]
        if record.comments[1] == 'True alarm':
            filedict[comments[0]][0].append(s[:-4])
        else:
            filedict[comments[0]][1].append(s[:-4])

    np.random.seed(seed)

    kf = KFold(n_splits=n_splits)

    test_files = [[] for _ in range(n_splits)]
    train_files = [[] for _ in range(n_splits)]
    for k, v in filedict.items():
        np.random.shuffle(v[0])
        np.random.shuffle(v[1])
        i = 0
        for train_index_positive, test_index_positive in kf.split(v[0]):
            test_files[i] = test_files[i] + list(np.array(v[0])[test_index_positive])
            train_files[i] = train_files[i] + list(np.array(v[0])[train_index_positive])
            i += 1
        i = 0
        for train_index_negative, test_index_negative in kf.split(v[1]):
            test_files[i] = test_files[i] + list(np.array(v[1])[test_index_negative])
            train_files[i] = train_files[i] + list(np.array(v[1])[train_index_negative])
            i += 1

    for i in range(n_splits):
        files_train, files_test = train_files[i], test_files[i]
        np.random.shuffle(files_train)
        np.random.shuffle(files_test)

        with open(os.path.join(origin_path, str(i + 1) + '_fold_test_files_list.txt'), 'w+') as test_file:
            for f in files_test:
                test_file.writelines(f + '\n')

        with open(os.path.join(origin_path, str(i + 1) + '_fold_train_files_list.txt'), 'w+') as train_file:
            for f in files_train:
                train_file.writelines(f + '\n')
    return


def resample_sig(cnt, point=1000):
    from scipy import signal
    cnt_resample = signal.resample(cnt, point, axis=0)
    return cnt_resample


def fill_nan(signal):
    """Solution provided by Divakar."""
    mask = np.isnan(signal)
    idx = np.where(~mask.T, np.arange(mask.shape[0]), 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    out = signal[idx.T, np.arange(idx.shape[0])[None, :]]
    out = np.nan_to_num(out)
    return out


def load_file_list(path, shuffle=True, seed=20200426):
    file_list = os.listdir(path)
    file_list = sorted(os.path.join(path, f[:-4]) for f in file_list if
                       os.path.isfile(os.path.join(path, f)) and f.endswith(".mat"))
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(file_list)
    return file_list


def load_file(filename, data_path):
    content = []
    with open(filename, 'r') as f:
        for line in f:
            content.append(os.path.join(data_path, line.strip('\n')))
    return content


def cross_val_files(data_folder, n_split):
    cv_datafiles = []
    for i in range(1, n_split + 1):
        data_path = defaultdict(list)
        data_path['train'] = load_file(data_folder + str(i) + '_fold_train_files_list.txt', data_folder)
        data_path['test'] = load_file(data_folder + str(i) + '_fold_test_files_list.txt', data_folder)
        cv_datafiles.append(data_path)
    return cv_datafiles


def load_record(filename):
    record = wfdb.rdrecord(filename)
    return record


def load_record_extra_info(filename):
    record = load_record(filename)
    fs = int(record.fs)
    sensor = record.sig_name
    event_classes = record.comments
    event_id = all_alarm_id[event_classes[0]]
    extra = event_id
    return extra


def get_chan_ind(sensor):
    chan_inds = [all_sensor_name.index(s) for s in sensor]
    return chan_inds


def load_full_sig(filename, fillnan=True):
    record = load_record(filename)
    fs = int(record.fs)
    continuous_signal = record.p_signal
    length = len(continuous_signal)
    cnt = np.full((length, len(all_sensor_name)), np.nan, dtype='float32')
    chan_inds = get_chan_ind(record.sig_name)
    cnt[:, chan_inds] = continuous_signal[:fs * length]
    if fillnan:
        cnt = fill_nan(cnt)
    event_classes = record.comments
    label = 0
    if event_classes[1] == 'True alarm':
        label = 1
    return cnt, label


def load_short_sig(filename, length=15, fillnan=True, gnorm=False):
    record = load_record(filename)
    fs = int(record.fs)
    cnt = np.full((fs * length, len(all_sensor_name)), np.nan, dtype='float32')
    continuous_signal = record.p_signal
    chan_inds = get_chan_ind(record.sig_name)
    cnt[:, chan_inds] = continuous_signal[(300 - length) * fs:300 * fs, :]

    if fillnan:
        cnt = fill_nan(cnt)
    if gnorm:
        tmp = np.full((fs * 10, len(all_sensor_name)), 0, dtype='float32')
        tmp[:, chan_inds] = continuous_signal[(290 - length) * fs:(300 - length) * fs, :]
        # minv = np.nanmin(tmp, axis=0)
        # minv = np.nan_to_num(minv)
        # maxv = np.nanmax(tmp, axis=0)
        # maxv = np.nan_to_num(maxv)
        # print(tmp.shape)
        minv = np.percentile(tmp, 5, axis=0)
        maxv = np.percentile(tmp, 95, axis=0)
        minv = np.nan_to_num(minv)
        maxv = np.nan_to_num(maxv)
        if isinstance(maxv, np.ndarray):
            t = [1 / v if v else 1 for v in (maxv - minv)]
        else:
            t = 1 / (maxv - minv) if maxv - minv else 1.0
        cnt -= minv
        cnt *= t

    event_classes = record.comments
    label = 0
    if event_classes[1] == 'True alarm':
        label = 1
    return np.array(cnt), label


def check_valid_channel(sig):
    sig = np.nan_to_num(sig)
    result = np.any(np.greater(abs(sig), 0), axis=0)
    #  or np.all((abs(sig) < 1e-12), axis=0) np.all(np.isnan(sig), axis=0) or
    return result


def te_check_valid_channel():
    sig = [[np.nan, np.nan],
           [0.0, 0.0], [1, 0.00000000000000000001], [np.nan, 0.0], [1, 0]]
    sig = np.array(sig)
    print(check_valid_channel(sig))


def load_short_need_sig(filename, length=15, fillnan=True, gnorm=False):
    record = load_record(filename)
    fs = int(record.fs)
    cnt = np.full((fs * length, 2), np.nan, dtype='float32')
    continuous_signal = record.p_signal
    i = 0
    if gnorm:
        tmp = np.full((fs * 10, 2), 0, dtype='float32')
    for j, s in enumerate(record.sig_name):
        if s in important_sig:
            cnt[:, i] = continuous_signal[(300 - length) * fs:300 * fs, j]
            if gnorm:
                tmp[:, i] = continuous_signal[(290 - length) * fs:(300 - length) * fs, j]
            i += 1
        if i == 2:
            break
    if i != 2:
        print(filename)
    if fillnan:
        cnt = fill_nan(cnt)
    if gnorm:
        # minv = np.nanmin(tmp, axis=0)
        # minv = np.nan_to_num(minv)
        # maxv = np.nanmax(tmp, axis=0)
        # maxv = np.nan_to_num(maxv)

        minv = np.percentile(tmp, 5, axis=0)
        maxv = np.percentile(tmp, 95, axis=0)
        minv = np.nan_to_num(minv)
        maxv = np.nan_to_num(maxv)
        if isinstance(maxv, np.ndarray):
            t = [1 / v if v else 1 for v in (maxv - minv)]
        else:
            t = 1 / (maxv - minv) if maxv - minv else 1.0
        cnt -= minv
        cnt *= t
    event_classes = record.comments
    label = 0
    if event_classes[1] == 'True alarm':
        label = 1
    return cnt, label


def minmax_scale(cnt):
    minv = np.nanmin(cnt, axis=0)
    maxv = np.nanmax(cnt, axis=0)
    if isinstance(maxv, np.ndarray):
        t = [1 / v if v else 1 for v in (maxv - minv)]
    else:
        t = 1 / (maxv - minv) if maxv - minv else 1.0
    cnt -= minv
    cnt *= t
    return cnt, minv, maxv


def wgn(cnt, snr):
    Ps = np.sum(abs(cnt) ** 2, axis=0) / len(cnt)
    Pn = Ps / (10 ** (snr / 10))
    Pn = np.reshape(Pn, (1, len(Pn)))
    noise = np.random.randn(*cnt.shape) * np.sqrt(Pn)
    cnt += noise
    return cnt


def sin_gaussion_noise(cnt, fz=50, factor='default', sigma='default'):
    # cnt = copy.deepcopy(sig)
    continer = [[i] * cnt.shape[1] for i in range(cnt.shape[0])]
    f = 0.1 * (np.nanmax(cnt, axis=0) - np.nanmin(cnt, axis=0))
    if factor == 'default':
        sin_noise = f * np.sin(np.array(continer) * fz * np.pi * 2 / 250 + np.random.randn(*cnt.shape) * np.pi)
    else:
        sin_noise = factor * np.sin(np.array(continer) * fz * np.pi * 2)
    if sigma == 'default':
        gauss_noise = f * np.random.randn(*cnt.shape)
    else:
        gauss_noise = sigma * np.random.randn(*cnt.shape)
    cnt += sin_noise + gauss_noise
    return cnt


def gaussion_noise(sig, sigma='default'):
    if sigma == 'default':
        sigma = 0.1 * (np.nanmax(sig, axis=0) - np.nanmin(sig, axis=0))
        noise = sigma * np.random.randn(*sig.shape)
    else:
        noise = sigma * np.random.randn(*sig.shape)
    sig += noise
    return sig


def amp_noise(sig):
    cnt = copy.deepcopy(sig)
    A = np.random.randn() + 0.5
    cnt = A * cnt
    return cnt


def sin_noise(sig, fz=50, factor='default'):
    cnt = copy.deepcopy(sig)
    continer = [[i] * cnt.shape[1] for i in range(cnt.shape[0])]
    if factor == 'default':
        factor = 0.5 * (np.nanmax(cnt, axis=0) - np.nanmin(cnt, axis=0))
        noise = factor * np.sin(np.array(continer) * fz * np.pi * 2)
    else:
        noise = factor * np.sin(np.array(continer) * fz * np.pi * 2)

    cnt += noise
    return cnt


def load_long_sig(filename, length=45, fillnan=True):
    record = load_record(filename)
    fs = int(record.fs)
    cnt = np.full((fs * length, len(all_sensor_name)), np.nan, dtype='float32')
    continuous_signal = record.p_signal
    chan_inds = get_chan_ind(record.sig_name)
    cnt[:, chan_inds] = continuous_signal[(330 - length) * fs:fs * 330]
    if fillnan:
        cnt = fill_nan(cnt)
    event_classes = record.comments
    label = 0
    if event_classes[1] == 'True alarm':
        label = 1
    return cnt, label


def sub_window_split(cnt, dilation=125, window=2500, start_point=0, mmscale=False, add_noise_prob=0.0,
                     add_amp_noise=False):
    result_cnt = []
    ch = cnt.shape[1]
    for i in range(start_point, cnt.shape[0], dilation):
        if i + window > cnt.shape[0]:
            break
        info = np.zeros((window, ch), dtype='float32')
        info[:] = cnt[i:i + window]
        if mmscale:
            minmax_scale(info)
        else:
            if add_amp_noise and np.random.uniform() > 0.5:
                amp_noise(info)
        if add_noise_prob > 0 and add_noise_prob > np.random.uniform():
            # p = np.random.uniform()
            # if p > gnoise:
            #     gaussion_noise(info)
            # else:
            #     sin_noise(info, 50)
            info = gaussion_noise(info)

        result_cnt.append(info)
    return np.array(result_cnt)


def filter_sensors(load_sensor_names=None):
    if load_sensor_names is None:
        return list(range(len(all_sensor_name)))
    chan_inds = get_chan_ind(load_sensor_names)
    return chan_inds


def stat_diff_alarm_nums(filelist):
    alarm_nums = defaultdict(list)
    for f in filelist:
        record = load_record(f)
        comments = record.comments
        if comments[0] not in alarm_nums:
            alarm_nums[comments[0]] = [0, 0]
        if comments[1] == 'True alarm':
            alarm_nums[comments[0]][1] += 1
        else:
            alarm_nums[comments[0]][0] += 1
    print(alarm_nums)


def chect_dataset():
    train_datapaths = ["1_fold_train_files_list.txt", "2_fold_train_files_list.txt", "3_fold_train_files_list.txt",
                       "4_fold_train_files_list.txt", "5_fold_train_files_list.txt"]
    test_datapaths = ["1_fold_test_files_list.txt", "2_fold_test_files_list.txt", "3_fold_test_files_list.txt",
                      "4_fold_test_files_list.txt", "5_fold_test_files_list.txt"]
    for i in range(5):
        train_files = load_file('../data/training/' + train_datapaths[i], '../data/training/')
        test_files = load_file('../data/training/' + test_datapaths[i], '../data/training/')
        print(len(train_files), len(test_files))
        files = set(train_files)
        for f in test_files:
            if f in files:
                print(i + 1, f)
        # files.update(set(test_files))
        # print(len(files))


if __name__ == "__main__":
    data_folder = '../data/training/'

    import wfdb

    # cross_val_files(data_folder, n_split=5)
    chect_dataset()

    # cross_val_dataset_balance(data_folder)
    exit(0)
    # split_dataset_balance(data_folder)

    test_file_list = load_file(data_folder + "test_files_list.txt", data_folder)
    stat_diff_alarm_nums(test_file_list)

    train_file_list = load_file(data_folder + "train_files_list.txt", data_folder)
    stat_diff_alarm_nums(train_file_list)

    all_file_list = load_file(data_folder + "RECORDS", data_folder)
    stat_diff_alarm_nums(all_file_list)

    # test_check_valid_channel()
    #
    root_path = '../data/training/'
    # print(stat_data(root_path))
    pid = 'a103l'
    # split_dataset(root_path)
    # split_dataset_balance(root_path)

    exit(0)
