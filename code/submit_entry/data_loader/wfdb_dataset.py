#!/usr/bin/python
# -*- coding: utf-8 -*-
from copy import deepcopy
from .data_read_utils import *


class DataLoader:

    def __init__(self, opt, testfile, alarm_type='all'):
        self.opt = opt
        self.testfile = testfile
        self.alarm_type = alarm_type
        self.test_sig_data = None
        self.test_sig_extra = None
        self.sig = None
        self.extra = None

    def _filter_sensors_data(self, extra, sensor_ind):
        row_index = []
        for i, e in enumerate(extra):
            valid_record = True
            for ind in sensor_ind:
                if e[0, ind] == 1:
                    valid_record = False
                    break
            if valid_record:
                row_index.append(i)
        return row_index

    def get_extro_len(self):
        return len(self.test_sig_extra)

    def clear_load_data(self):
        self.test_sig_data = None
        self.test_sig_extra = None
        self.sig = None
        self.extra = None

    def load_test_data(self, clear=False):
        if clear:
            self.clear_load_data()
        wanted_chan_inds = filter_sensors(self.opt.load_sensor_names)
        if self.opt.load_important_sig:
            cnt = load_short_need_sig(self.testfile, length=self.opt.load_sig_length, gnorm=self.opt.use_global_minmax)
            self.test_sig_data = deepcopy(cnt)
        else:
            cnt = load_short_sig(self.testfile, length=self.opt.load_sig_length, gnorm=self.opt.use_global_minmax)
            self.test_sig_data = cnt[:, wanted_chan_inds]
        self.test_sig_extra = load_record_extra_info(self.testfile)
        return

    def load_data(self, check=True):
        self.load_test_data(True)
        cnt, extra = self.test_sig_data, self.test_sig_extra
        valid = check_valid_channel(cnt)
        result_cnt = sub_window_split(cnt, dilation=self.opt.dilation, window=self.opt.window_size,
                                      mmscale=self.opt.use_minmax_scale,
                                      add_noise_prob=0,
                                      gnoise=self.opt.gaussian_noise,
                                      snoise=self.opt.sin_noise)
        n = len(result_cnt)
        result_extra = [extra[:] for _ in range(n)]
        if self.opt.load_important_sig and check:
            if np.all(valid):
                result_cnt = [result_cnt[:, :, 0][:, :, np.newaxis], result_cnt[:, :, 1][:, :, np.newaxis]]
            elif valid[0]:
                result_cnt = result_cnt[:, :, 0][:, :, np.newaxis]
            elif valid[1]:
                result_cnt = result_cnt[:, :, 1][:, :, np.newaxis]
            else:
                return
        if isinstance(result_cnt, list):
            sig = []
            for cnt in result_cnt:
                sig += [np.transpose(cnt, (0, 2, 1))]
        else:
            sig = np.transpose(result_cnt, (0, 2, 1))
        self.sig = sig
        self.extra = np.array(result_extra) if self.opt.use_extra else None
        return

    def check_file_valid(self, alarm_type='all', explict_sensor=False, need_long_record=False):

        if need_long_record and self.testfile.endswith('s'):
            return False

        record = wfdb.rdrecord(self.testfile)
        if alarm_type != 'all' and record.comments[0] != alarm_type:
            return False

        sensors = {k: v for v, k in enumerate(record.sig_name)}
        check = check_valid_channel(record.p_signal[-20 * 250:])
        have_all_need_sensor = True
        if explict_sensor:
            for s in self.opt.load_sensor_names:
                if s not in sensors or not check[sensors[s]]:
                    have_all_need_sensor = False
                    break

        if self.opt.load_important_sig:
            for s, ind in sensors.items():
                if s in important_sig and check[ind]:
                    have_all_need_sensor = True
                    break
        return have_all_need_sensor
