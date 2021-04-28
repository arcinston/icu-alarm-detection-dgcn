from abc import ABC, abstractmethod
import logging
import time
import prettytable as pt
import csv

log = logging.getLogger(__name__)


class Logger(ABC):
    @abstractmethod
    def log_epoch(self, epochs_df):
        raise NotImplementedError("Need to implement the log_epoch function!")


class Printer(Logger):
    """
    Prints output to the terminal using Python's logging module.
    """

    def log_epoch(self, epochs_df):
        # -1 due to doing one monitor at start of training
        i_epoch = len(epochs_df) - 1
        log.info("Epoch {:d}".format(i_epoch))
        last_row = epochs_df.iloc[-1]
        for key, val in last_row.iteritems():
            log.info("{:25s} {:.5f}".format(key, val))
        log.info("")


class PrinterTable(Logger):
    """
    Prints output to the terminal using Python's logging module.
    """

    def __init__(self, file_path):
        self.file_path = file_path
        self.header = dict()
        self.row = dict()
        self.data = None

    def init_table(self, epochs_df):
        last_row = epochs_df.iloc[-1]
        h_id, r_id = 2, 1
        self.header['Alarm'] = 0
        self.header["Set"] = 1
        for key, val in last_row.iteritems():
            labels = key.split('_')
            if (labels[0], labels[1]) not in self.row:
                self.row[(labels[0], 'train')] = r_id
                self.row[(labels[0], 'test')] = r_id + 1
                r_id += 2
            if labels[2] not in self.header:
                self.header[labels[2]] = h_id
                h_id += 1

        self.data = [['--'] * len(self.header) for _ in range(len(self.row) + 1)]
        for k, v in self.header.items():
            self.data[0][v] = k

        for k, v in self.row.items():
            self.data[v][0] = k[0]
            self.data[v][1] = k[1]

    def empty(self):
        assert self.data is not None
        m = len(self.data)
        n = len(self.data[0])
        for i in range(1, m):
            for j in range(2, n):
                self.data[i][j] = '--'
        return

    def log_epoch(self, epochs_df):
        if self.data is None:
            self.init_table(epochs_df)
        else:
            self.empty()

        # -1 due to doing one monitor at start of training
        i_epoch = len(epochs_df) - 1
        log.info("Epoch {:d}".format(i_epoch))
        last_row = epochs_df.iloc[-1]
        for key, val in last_row.iteritems():
            if isinstance(val, float):
                val = round(val, 4)
            labels = key.split('_')
            h_id, r_id = self.header[labels[2]], self.row[(labels[0], labels[1])]
            self.data[r_id][h_id] = val

        for k, r_id in self.row.items():
            for h in ['TP', 'TN', 'FP', 'FN', 'Total']:
                h_id = self.header[h]
                self.data[r_id][h_id] = int(self.data[r_id][h_id])

        tb = pt.PrettyTable()

        tb.field_names = self.data[0]
        for i in range(1, len(self.row) + 1):
            tb.add_row(self.data[i])

        print(tb)
        log.info("")
        with open(self.file_path, 'a+') as log_file:
            log_file.writelines("Epoch {:d}".format(i_epoch) + '\n')
            log_file.write(str(tb) + '\n')
            log_file.writelines('\n')
        return


class PrinterCSVFile(Logger):
    """
    Prints output to the terminal using Python's logging module.
    """

    def __init__(self, file_path):
        self.file_path = file_path
        self.header = dict()
        self.row = dict()
        self.data = None

    def init_table(self, epochs_df):
        last_row = epochs_df.iloc[-1]
        h_id, r_id = 3, 1
        self.header['Epoch'] = 0
        self.header['Alarm'] = 1
        self.header["Set"] = 2
        for key, val in last_row.iteritems():
            labels = key.split('_')
            if (labels[0], labels[1]) not in self.row:
                self.row[(labels[0], 'train')] = r_id
                self.row[(labels[0], 'test')] = r_id + 1
                r_id += 2
            if labels[2] not in self.header:
                self.header[labels[2]] = h_id
                h_id += 1

        self.data = [['--'] * len(self.header) for _ in range(len(self.row) + 1)]
        for k, v in self.header.items():
            self.data[0][v] = k

        for k, v in self.row.items():
            self.data[v][0] = k[0]
            self.data[v][1] = k[1]

    def empty(self):
        assert self.data is not None
        m = len(self.data)
        n = len(self.data[0])
        for i in range(1, m):
            for j in range(3, n):
                self.data[i][j] = '--'
        return

    def log_epoch(self, epochs_df):
        if self.data is None:
            self.init_table(epochs_df)
        else:
            self.empty()

        # -1 due to doing one monitor at start of training
        i_epoch = len(epochs_df) - 1
        log.info("Epoch {:d}".format(i_epoch))
        last_row = epochs_df.iloc[-1]
        for key, val in last_row.iteritems():
            if isinstance(val, float):
                val = round(val, 4)
            labels = key.split('_')
            h_id, r_id = self.header[labels[2]], self.row[(labels[0], labels[1])]
            self.data[r_id][h_id] = val

        h3_id = self.header['Epoch']
        for k, r_id in self.row.items():
            for h in ['TP', 'TN', 'FP', 'FN', 'Total']:
                h_id = self.header[h]
                self.data[r_id][h_id] = int(self.data[r_id][h_id])
            self.data[r_id][h3_id] = i_epoch

        # -1 due to doing one monitor at start of training
        with open(self.file_path, 'a+') as log_file:
            writer = csv.writer(log_file)
            writer.writerows(self.data)
            writer.writerows([])
        return


class TensorboardWriter(Logger):
    """
    Logs all values for tensorboard visualiuzation using tensorboardX.

    Parameters
    ----------
    log_dir: string
        Directory path to log the output to
    """

    def __init__(self, log_dir):
        # import inside to prevent dependency of Ecgdecode onto tensorboardX
        from torch.utils.tensorboard import SummaryWriter
        self.time_str = time.strftime("%Y-%m-%d-%H-%M-%S/", time.localtime())
        self.test_writer = SummaryWriter(log_dir + 'test_' + self.time_str)
        self.valid_writer = SummaryWriter(log_dir + 'valid_' + self.time_str)
        self.train_writer = SummaryWriter(log_dir + 'train_' + self.time_str)

    def log_epoch(self, epochs_df):
        # -1 due to doing one monitor at start of training
        i_epoch = len(epochs_df) - 1
        last_row = epochs_df.iloc[-1]
        for key, val in last_row.iteritems():
            info = key.split('_')
            val = last_row[key]
            if info[0] == 'test':
                self.test_writer.add_scalar(info[1], val, i_epoch)
            elif info[0] == 'train':
                self.train_writer.add_scalar(info[1], val, i_epoch)
            elif info[0] == 'valid':
                self.valid_writer.add_scalar(info[1], val, i_epoch)
