class Opt:
    def __init__(self):
        self.base_setting()
        self.optimizer_setting()
        self.data_setting()
        self.signal_setting()
        self.setting_of_deep_modified()
        return

    def base_setting(self):
        self.batch_size = 64
        self.load_people_size = 64
        self.input_nc = 2
        self.input_length = 2500
        self.lr = 1e-4
        self.max_epoch = 400
        self.max_increase_epoch = 80
        self.use_cross_val = False
        self.np_to_seed = 1024  # random seed for numpy and pytorch
        self.model_name = 'dgcn'  # /deep_modified_embedded

        self.debug = False
        self.training = True
        self.cuda = True

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

    def optimizer_setting(self):
        self.lr = 1e-4
        self.weight_decay = 0

    def data_setting(self):
        # self.data_folder = './data/training/'
        self.data_folder = '../data/training/'
        self.train_files = self.data_folder + 'train_files_list.txt'
        self.test_files = self.data_folder + 'test_files_list.txt'
        self.train_folder = '../data/train/'
        self.test_folder = '../data/test/'
        self.train_file_base_name = '../data/train'
        self.valid_file_base_name = '../data/valid'
        self.test_file_base_name = '../data/test'
        self.all_file_base_name = '../data/allrecord'
        self.gen_new_npy = False
        self.split = True
        self.data_folder = '../data/training/'

        self.train_files = self.data_folder + 'train_files_list.txt'
        self.test_files = self.data_folder + 'test_files_list.txt'
        self.all_file_base_name = '../data/allrecord'
        self.train_datapaths = ["1_fold_train_files_list.txt", "2_fold_train_files_list.txt",
                                "3_fold_train_files_list.txt",
                                "4_fold_train_files_list.txt", "5_fold_train_files_list.txt"]

        self.test_datapaths = ["1_fold_test_files_list.txt", "2_fold_test_files_list.txt", "3_fold_test_files_list.txt",
                               "4_fold_test_files_list.txt", "5_fold_test_files_list.txt"]
        self.fold = 5
        self.sub_names = ['PRA.pth', 'II.pth', 'V.pth', 'ECG.pth']

    def signal_setting(self):
        self.low_cut_hz = 0  # or 4
        self.use_extra = False
        self.extra_length = 5
        self.n_split = 5
        self.SECOND_LENGTH = 300
        self.LONG_SECOND_LENGTH = 450
        self.load_sig_length = 15
        self.use_minmax_scale = False
        self.use_global_minmax = False
        self.use_gaussian_noise = False
        self.use_amplitude_noise = False
        self.load_important_sig = False
        self.add_noise_prob = 0.8
        self.gaussian_noise = 0.5
        self.sin_noise = 0.5
        self.gaussian_snr = 50
        self.test_size = 0.1
        self.valid_size = 0.2
        self.load_sensor_names = ['II', 'V']
        self.window_size = 2500
        self.dilation = int(0.5 * 250)

        # load_sensor_names = ['PLETH', 'RESP', 'ABP']
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
