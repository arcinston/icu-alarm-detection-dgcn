import sys
import os.path
import numpy as np

import wfdb
import torch
from models.dgcn import DGCN

all_sensor_name = ['I', 'II', 'III', 'V', 'aVL', 'aVR', 'aVF', 'RESP', 'PLETH', 'MCL', 'ABP']
load_sensor_names = ['II', 'V', 'RESP', 'PLETH', 'ABP']


def fill_nan(signal):
    '''Solution provided by Divakar.'''
    mask = np.isnan(signal)
    idx = np.where(~mask.T, np.arange(mask.shape[0]), 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    out = signal[idx.T, np.arange(idx.shape[0])[None, :]]
    return out


def load_data(filename):
    record = wfdb.rdrecord(filename)
    # 获取信号采样率
    fs = int(record.fs)
    sensor = record.sig_name
    SECOND_LENGTH = 15
    cnt = np.full((fs * SECOND_LENGTH, len(load_sensor_names)), np.nan, dtype='float32')
    continuous_signal = record.p_signal
    chan_inds = [load_sensor_names.index(s) for s in sensor]
    cnt[:, chan_inds] = continuous_signal[(300 - SECOND_LENGTH) * fs:300 * fs, :]
    cnt = fill_nan(cnt)
    cnt = np.nan_to_num(cnt)
    cnt = cnt.transpose(1, 0)
    cnt = cnt[np.newaxis, :]
    print(cnt.shape)
    X_tensor = torch.tensor(cnt, requires_grad=False, dtype=torch.float32)
    print(X_tensor.size())
    return X_tensor


def predict(model, input):
    model.eval()
    with torch.no_grad():
        outputs = model(input)
        outputs = outputs.cpu().detach().numpy()

    if outputs[0] >= 0.5:
        return 1
    return 0


n_chans = 5
input_time_length = 15 * 250


def load_model(path):
    import torch
    model = DGCN(n_chans, 2,
                 input_time_length=input_time_length,
                 n_filters_time=15,
                 final_conv_length="auto", drop_prob=0.3, stride_before_pool=False)
    model.load_state_dict(torch.load(path))
    return model


if __name__ == "__main__":
    load_data('./data/training/training/a103l')
    model_path = './checkpoints/latest/all_4_latest.pth'
    model = load_model(model_path)
    fp = open("answers.txt", "a+", encoding="utf-8")
    for record in sys.argv[1:]:
        output_file = os.path.basename(record)
        data = load_data(record)
        results = model.predict(data)
        fp.write(output_file + "," + str(results) + "\n")
    fp.close()
