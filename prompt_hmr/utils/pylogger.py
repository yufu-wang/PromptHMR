import os
import shutil
import logging
from pytorch_lightning.utilities import rank_zero_only


def get_pylogger(name=__name__) -> logging.Logger:
    """Initializes multi-GPU-friendly python command line logger."""

    # log_file = os.path.join(f'test_log.txt')
    # head = '%(asctime)-15s %(message)s'
    # logging.basicConfig(filename=log_file,
    #                     format=head)
    
    # logger = logging.getLogger(name)
    # console = logging.StreamHandler()
    # logging.getLogger('').addHandler(console)


    logger = logging.getLogger(name)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    logging_levels = ("debug", "info", "warning", "error", "exception", "fatal", "critical")
    for level in logging_levels:
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger


def create_logger(logdir, phase='train'):
    os.makedirs(logdir, exist_ok=True)

    log_file = os.path.join(logdir, f'{phase}_log.txt')

    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=log_file,
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    return logger


def prepare_output_dir(cfg):

    if not cfg.LOGDIR:
        logfolder = cfg.EXP_NAME
        logdir    = os.path.join(cfg.OUTPUT_DIR, logfolder)
        os.makedirs(logdir, exist_ok=True)

        cfg.LOGDIR = logdir

    shutil.copy(src=cfg.cfg_file, dst=cfg.LOGDIR + '/config.yaml')

    return cfg