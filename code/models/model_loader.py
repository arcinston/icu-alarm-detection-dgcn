from .dgcn import DGCN
from .edgcn import EDGCN
from .deeper_edgcn import DeeperEDGCN
from .deeper_dgcn import DeeperDGCN


def load_model(opt=None):
    assert opt is not None
    model = None
    model_name = opt.model_name
    if model_name == "dgcn":
        model = DGCN(opt)
    elif model_name == "deeper":
        model = DeeperDGCN(opt)
    elif model_name == "deeper_edgcn":
        model = DeeperEDGCN(opt)
    elif model_name == "edgcn":
        opt.use_extra = True
        model = EDGCN(opt.input_nc, n_classes=opt.n_classes,
                      input_time_length=opt.input_length,
                      extra_length=opt.extra_length,
                      drop_prob=opt.drop_prob)

    else:
        raise ValueError("Unknown Model Name!!!")

    if not model:
        raise ValueError("Load Model Error!!!")
    if opt.cuda:
        model.cuda()
    print(model)
    model.eval()
    return model
