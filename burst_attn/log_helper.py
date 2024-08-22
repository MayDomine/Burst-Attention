import logging
def get_logger(name, level="INFO", log_file=None):
    logger = logging.getLogger(name)
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] "
        "[%(filename)s:%(lineno)d:%(funcName)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    if log_file:
        handler = logging.FileHandler(log_file)
    else:
        handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(level)
    return logger
