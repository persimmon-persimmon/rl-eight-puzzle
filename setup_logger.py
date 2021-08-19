import logging
def setup_logger(name,logfile = None, filesize = 1024 * 1024 * 128, backupCount = 5):
    """
    loggerインスタンスを返す.
    logfileを指定すればログをファイル出力する.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt="%(asctime)s - %(levelname)s - %(message)s",datefmt="%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    sh.setLevel(logging.INFO)
    logger.addHandler(sh)
    if logfile is not None:
        fh = logging.RotatingFileHandler(filename=logfile, maxBytes=filesize, backupCount=backupCount)
        fh.setFormatter(formatter)
        fh.setLevel(logging.INFO)
        logger.addHandler(fh)
    return logger
