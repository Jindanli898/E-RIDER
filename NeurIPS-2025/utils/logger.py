import logging
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


class Logger(object):
    """
    Helper class for logging.
    Writes logs to console and (optionally) to a file colocated with TensorBoard events.

    Args:
        log_dir (str): Directory for TensorBoard event files (and the paired .log).
        log_in_file (bool): Whether to log to a .log file in addition to console.
    """
    def __init__(self, log_dir, log_in_file=True):
        self.log_in_file = log_in_file
        self.log_dir = log_dir

        if log_in_file:
            self.writer = SummaryWriter(log_dir=log_dir)
            # Full path to TB event file (without ".log" suffix).
            self.event_write_file_name = self.writer.file_writer.event_writer._file_name  # e.g., .../events.out.tfevents...
            print(f"Logging to file: {self.event_write_file_name}")
        else:
            self.writer = None
            self.event_write_file_name = None

        self.logger = self._get_logger(log_dir, self.event_write_file_name, log_in_file)

    def get_text_log_path(self):
        """
        Return the full path to the paired text log file (events... .log).
        Useful for naming your config JSON consistently.
        """
        if not self.log_in_file or self.event_write_file_name is None:
            return None
        return self.event_write_file_name + ".log"

    def get_text_log_basename(self):
        """
        Return only the filename of the paired text log file.
        """
        p = self.get_text_log_path()
        return None if p is None else Path(p).name

    def _get_logger(self, log_dir, event_write_file_name, log_in_file=True):
        # If need to log in file, the ``event_write_file_name`` should not be None.
        assert event_write_file_name is not None or not log_in_file

        logger = logging.getLogger(log_dir)

        # Prevent duplicate handlers if Logger(log_dir) is constructed multiple times.
        if logger.handlers:
            return logger

        # Set up the logger.
        logger.setLevel(logging.DEBUG)

        # Define the log format.
        format_str = "[%(asctime)s] %(message)s"
        formatter = logging.Formatter(format_str)

        # Console handler.
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File handler.
        if log_in_file:
            file_handler = logging.FileHandler(event_write_file_name + ".log")
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        return logger

    def write(self, step, message, tb_dict):
        # TensorBoard record.
        if self.log_in_file and self.writer is not None:
            for key, value in tb_dict.items():
                self.writer.add_scalar(key, value, step)

        # Terminal / file record.
        if message is not None:
            self.logger.info(message)
