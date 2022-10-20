import sys
import logging
from logging.handlers import RotatingFileHandler
import matplotlib.pyplot as plt


def plot_losses(train_losses, valid_losses, path):
    plt.plot(train_losses, label='train')
    plt.plot(valid_losses, label='valid')
    plt.savefig(f"{path}/losses.png")
    plt.clf()


class CustomLogger:

    def __init__(self,
                 name,
                 file_path=None,
                 log_size=10 * 1024 * 1024,
                 backup_count=5):
        self.log_size = log_size
        self.backup_count = backup_count
        self._init_logger(name, file_path)

    def log_info(self, message):
        self.logger.info(message)

    def _init_logger(self, name, file_path):
        logging.addLevelName(logging.INFO, "[INF]")

        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        self.formatter = logging.Formatter(
            "%(levelname)s - %(asctime)s - %(message)s")

        if file_path:
            file_handler = RotatingFileHandler(file_path,
                                               maxBytes=self.log_size,
                                               backupCount=self.backup_count)
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(self.formatter)
            self.logger.addHandler(file_handler)
        else:
            stream_handler = logging.StreamHandler(sys.stdout)
            stream_handler.setFormatter(self.formatter)
            self.logger.addHandler(stream_handler)
