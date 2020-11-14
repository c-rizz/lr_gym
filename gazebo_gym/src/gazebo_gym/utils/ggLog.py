#!/usr/bin/env python3

import os
import logging


logger = logging.getLogger('GGLog')
logger.setLevel(logging.DEBUG)
# create file handler that logs debug and higher level messages
fh = logging.FileHandler('spam.log')
fh.setLevel(logging.DEBUG)
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
# create formatter and add it to the handlers
formatter = logging.Formatter('[%(asctime)s.%(msecs)03d][%(levelname)s] %(message)s', datefmt='%Y%m%d,%H:%M:%S')
ch.setFormatter(formatter)
fh.setFormatter(formatter)
# add the handlers to logger
logger.addHandler(ch)
logger.addHandler(fh)

def _addRosMasterUri(msg):
    ros_master_uri = os.environ['ROS_MASTER_URI'].split(":")[-1]
    if ros_master_uri is None:
        return "[] "+msg
    else:
        return "["+str(ros_master_uri)+"] "+msg

def info(msg, *args, **kwargs):
    logger.info(_addRosMasterUri(msg), *args, **kwargs)

def debug(msg, *args, **kwargs):
    logger.debug(_addRosMasterUri(msg), *args, **kwargs)

def warn(msg, *args, **kwargs):
    logger.warn(_addRosMasterUri(msg), *args, **kwargs)

def error(msg, *args, **kwargs):
    logger.error(_addRosMasterUri(msg), *args, **kwargs)

def critical(msg, *args, **kwargs):
    logger.critical(_addRosMasterUri(msg), *args, **kwargs)

def exception(msg, *args, **kwargs):
    logger.exception(_addRosMasterUri(msg), *args, **kwargs)
