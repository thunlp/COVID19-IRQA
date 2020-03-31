import sys
import logging


def set_logger(log_file):
    """ create logging file. """
    logger = logging.getLogger() 
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler() 
    console.setFormatter(fmt) 

    logfile = logging.FileHandler(log_file, 'w')
    logfile.setFormatter(fmt)
    logger.addHandler(logfile)
    logger.info('COMMAND: %s' % ' '.join(sys.argv))