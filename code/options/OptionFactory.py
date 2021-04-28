from options.default_opt import Opt


def gen_options(model='edgcn', use_slice=True, use_norm=False, use_noise=False, use_gnorm=False):
    opt = Opt()
    opt.drop_prob = 0.3
    opt.load_sig_length = 15
    if model == 'dgcn':
        opt.model_name = 'dgcn'
        opt.use_extra = False
    elif model == 'edgcn':
        opt.model_name = 'edgcn'
        opt.use_extra = True
    elif model == 'deeper':
        opt.model_name = 'deeper_edgcn'
        opt.use_extra = True
    if use_slice:
        opt.load_sig_length = 15
        opt.window_size = 3000
        opt.input_length = 3000
    else:
        opt.load_sig_length = 15
        opt.window_size = 3750
        opt.input_length = 3750

    if use_norm:
        opt.use_minmax_scale = True
    else:
        opt.use_minmax_scale = False

    if use_gnorm:
        opt.use_global_minmax = True
    else:
        opt.use_global_minmax = False

    if use_noise:
        opt.add_noise_prob = 0.8
    else:
        opt.add_noise_prob = 0

    return opt
