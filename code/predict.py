import sys
import os.path
import numpy as np

import wfdb
import torch
from models.dgcn import DGCN

all_sensor_name = ['I', 'II', 'III', 'V', 'aVL', 'aVR', 'aVF', 'RESP', 'PLETH', 'MCL', 'ABP']
load_sensor_names = ['II', 'V', 'RESP', 'PLETH', 'ABP']


def fill_nan(signal):
    mask = np.isnan(signal)
    idx = np.where(~mask.T, np.arange(mask.shape[0]), 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    out = signal[idx.T, np.arange(idx.shape[0])[None, :]]
    return out


def load_data(filename):
    record = wfdb.rdrecord(filename)
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
    model = DGCN()
    model.load_state_dict(torch.load(path))
    return model


def prepare_input(input_data):
    # Scale the input data to the range [0, 1]
    input_data = (input_data - np.min(input_data)) / (np.max(input_data) - np.min(input_data))
    # Convert the input data to a PyTorch tensor
    input_tensor = torch.from_numpy(input_data).float()
    # Add a batch dimension to the tensor
    input_tensor = input_tensor.unsqueeze(0)
    return input_tensor

if __name__ == "__main__":
    data_path = './data/training/training/a103l'
    model_path = './checkpoints/latest/best.pth'
    model = load_model(model_path)
    data = load_data(data_path)
    fp = open("answers.txt", "a+", encoding="utf-8")
    for record in sys.argv[1:]:
        output_file = os.path.basename(record)
        input_data = load_data(record)
        input_tensor = prepare_input(input_data)
        results = predict(model, input_tensor)
        fp.write(output_file + "," + str(results) + "\n")
    fp.close()