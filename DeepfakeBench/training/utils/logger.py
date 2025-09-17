import os
import sys
import logging

import torch.distributed as dist

class RankFilter(logging.Filter):
    def __init__(self, rank):
        super().__init__()
        self.rank = rank

    def filter(self, record):
        return dist.get_rank() == self.rank


def create_logger(log_path):
    # Create log path
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    # Create logger object
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Remove existing handlers to prevent duplicate logging
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create file handler and set the formatter
    fh = logging.FileHandler(log_path, mode='a')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Add a stream handler to print to console
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)  # Set logging level for stream handler
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

