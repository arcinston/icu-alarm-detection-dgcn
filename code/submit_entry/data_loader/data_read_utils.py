#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import sys
from sys import exit
import logging
import numpy as np
import shutil
import wfdb
import re
from collections import defaultdict

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

all_15s = {'Tachycardia', 'Ventricular_Flutter_Fib', 'Asystole'}


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


def load_record(filename):
    record = wfdb.rdrecord(filename)
    return record


all_alarm = {'Ventricular_Tachycardia': 'VTA', 'Tachycardia': 'ETC',
             'Ventricular_Flutter_Fib': 'VFB', 'Bradycardia': 'EBR',
             'Asystole': 'ASY'}


def get_record_label(filename):
    record = wfdb.rdrecord(filename)
    label = 0
    if record.comments[1] == 'True alarm':
        label = 1
    return all_alarm[record.comments[0]], label


def load_record_extra_info(filename):
    record = load_record(filename)
    fs = int(record.fs)
    sensor = record.sig_name
    # sig_nan_ch = []
    # for i, s in enumerate(all_sensor_name):
    #     if s in sensor:
    #         sig_nan_ch.append(0)
    #     else:
    #         sig_nan_ch.append(1)
    event_classes = record.comments
    event_id = all_alarm_id[event_classes[0]]
    # extra = [fs] + sig_nan_ch + event_id
    extra = event_id
    return extra


def load_record_use_15s(filename):
    record = load_record(filename)
    event_classes = record.comments
    if event_classes[0] in all_15s:
        return True
    return False


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

    return np.array(cnt)


def load_short_need_sig(filename, length=15, fillnan=True, gnorm=False):
    record = load_record(filename)
    fs = int(record.fs)
    cnt = np.full((fs * length, 2), np.nan, dtype='float32')
    continuous_signal = record.p_signal
    chan_inds = get_chan_ind(record.sig_name)
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

    if fillnan:
        cnt = fill_nan(cnt)
    if gnorm:
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
    return np.array(cnt)


def check_valid_channel(sig):
    sig = np.nan_to_num(sig)
    result = np.any(np.greater(abs(sig), 0), axis=0)
    return result


def minmax_scale(cnt):
    minv = np.nanmin(cnt, axis=0)
    maxv = np.nanmax(cnt, axis=0)
    if isinstance(maxv, np.ndarray):
        t = [1 / v if v else 1 for v in (maxv - minv)]
    else:
        t = 1 / (maxv - minv) if maxv - minv else 1
    cnt -= minv
    cnt *= t
    return cnt, minv, maxv


def wgn(cnt, snr):
    Ps = np.sum(abs(cnt) ** 2, axis=0) / len(cnt)
    Pn = Ps / (10 ** ((snr / 10)))
    Pn = np.reshape(Pn, (1, len(Pn)))
    noise = np.random.randn(*cnt.shape) * np.sqrt(Pn)
    cnt += noise
    return cnt


def gaussion_noise(cnt, sigma='default'):
    if sigma == 'default':
        sigma = 0.1 * (np.nanmax(cnt, axis=0) - np.nanmin(cnt, axis=0))
        noise = sigma * np.random.randn(*cnt.shape)
    else:
        noise = sigma * np.random.randn(*cnt.shape)
    cnt += noise
    return cnt


def sin_noise(cnt, fz=50, factor='default'):
    continer = [[i] * cnt.shape[1] for i in range(cnt.shape[0])]
    if factor == 'default':
        factor = 0.1 * (np.nanmax(cnt, axis=0) - np.nanmin(cnt, axis=0))
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


def sub_window_split(cnt, dilation=125, window=2500, start_point=0, mmscale=False, add_noise_prob=0.0, gnoise=0.0,
                     snoise=0.0):
    result_cnt = []
    ch = cnt.shape[1]
    for i in range(start_point, cnt.shape[0], dilation):
        if i + window > cnt.shape[0]:
            break
        info = np.zeros((window, ch), dtype='float32')
        info[:] = cnt[i:i + window]
        if mmscale:
            minmax_scale(info)
        if add_noise_prob > 0 and add_noise_prob > np.random.uniform():
            gaussion_noise(info)

        result_cnt.append(info)
    return np.array(result_cnt)


def filter_sensors(load_sensor_names=None):
    if load_sensor_names is None:
        return list(range(len(all_sensor_name)))
    chan_inds = get_chan_ind(load_sensor_names)
    return chan_inds
