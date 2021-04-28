#!/usr/bin/python
# -*- coding: utf-8 -*-
class Opt:
    def __init__(self):
        self.base_setting()
        self.signal_setting()
        self.setting_of_deep_modified()
        # self.setting_of_deep_embedded()
        return

    def base_setting(self):
        self.batch_size = 64
        self.input_nc = 2
        self.input_length = 3000
        self.model_name = 'dgcn'  # /deep_modified_embedded
        self.cuda = False

    def setting_of_deep_modified(self):
        self.input_nc = 2
        self.n_classes = 2
        self.input_length = 2500
        self.n_filters = 15
        self.drop_prob = 0.1
        self.stride_before_pool = False
        self.filter_length = 10
        self.channel_res_conv_length = 3
        self.pool_length = 3
        self.pool_stride = 3
        self.n_filters_2 = 15
        self.filter_length_2 = 10
        self.n_filters_3 = 30
        self.filter_length_3 = 10
        self.n_filters_4 = 30
        self.filter_length_4 = 10
        self.batch_norm = True
        self.split = True

    def setting_of_deep_embedded(self):
        self.input_nc = 2
        self.n_classes = 2
        self.input_length = 3000
        self.n_filters = 15
        self.drop_prob = 0.1
        self.filter_length = 10
        self.n_filters_2 = 15
        self.filter_length_2 = 10
        self.n_filters_3 = 30
        self.filter_length_3 = 10
        self.n_filters_4 = 30
        self.filter_length_4 = 10
        self.batch_norm = True

    def signal_setting(self):
        self.low_cut_hz = 0  # or 4
        self.use_extra = True
        self.extra_length = 5
        self.SECOND_LENGTH = 300
        self.LONG_SECOND_LENGTH = 450
        self.use_minmax_scale = True
        self.use_global_minmax = False
        self.load_important_sig = False
        self.add_noise_prob = 0
        self.gaussian_noise = 0.5
        self.sin_noise = 0.5
        self.gaussian_snr = 50
        self.window_size = 3000
        self.dilation = int(0.5 * 250)
        self.load_sig_length = 15
        self.load_sensor_names = ['II', 'V']

        # I	    II	 III	V	    aVL 	aVR	    aVF	    RESP  PLETH	   MCL	   ABP
        # mV	mV	 mV	    mV	    mV	    mV	    mV	    NU	  NU	   mV	   mmHg
        # 13	728	 39	    684	    2	    3	    3	    278	  627	   28	   343
        # 1.7%	97%	 5.2%	91.2%	0.27%	0.4%	0.4%	37%	  83.6%	   3.7%	   45.7%

        self.all_sensor_unit = {'II': 'mV', 'V': 'mV', 'PLETH': 'NU', 'aVF': 'mV', 'ABP': 'mmHg', 'RESP': 'NU',
                                'III': 'mV', 'MCL': 'mV', 'I': 'mV', 'aVR': 'mV', 'aVL': 'mV'}
        self.all_sensor_name = ['I', 'II', 'III', 'V', 'aVL', 'aVR', 'aVF', 'RESP', 'PLETH', 'MCL', 'ABP']
        self.all_alarm_type = ['Ventricular_Tachycardia', 'Tachycardia', 'Ventricular_Flutter_Fib', 'Bradycardia',
                               'Asystole']
        self.all_alarm_id = {'Ventricular_Tachycardia': [1, 0, 0, 0, 0], 'Tachycardia': [0, 1, 0, 0, 0],
                             'Ventricular_Flutter_Fib': [0, 0, 1, 0, 0], 'Bradycardia': [0, 0, 0, 1, 0],
                             'Asystole': [0, 0, 0, 0, 1]}
        self.all_alarm_error_count = {'Asystole': [100, 22], 'Bradycardia': [43, 46],
                                      'Ventricular_Flutter_Fib': [52, 6], 'Tachycardia': [9, 131],
                                      'Ventricular_Tachycardia': [252, 89]}

    def __repr__(self):
        return '\n'.join(['%s:%s' % item for item in self.__dict__.items()])
