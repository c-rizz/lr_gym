#!/usr/bin/env python3

import os
import logging
import datetime


logger = logging.getLogger('GGLog')
logger.setLevel(logging.DEBUG)
# create file handler that logs debug and higher level messages
os.makedirs("ggLogs", exist_ok=True)
fh = logging.FileHandler('ggLogs/ggLog_'+datetime.datetime.now().strftime('%Y%m%d-%H%M%S')+'.log')
fh.setLevel(logging.DEBUG)
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
# create formatter and add it to the handlers
formatter = logging.Formatter('[%(asctime)s.%(msecs)03d][%(levelname)s] %(message)s', datefmt='%Y%m%d,%H:%M:%S')
ch.setFormatter(formatter)
fh.setFormatter(formatter)
# add the handlers to logger
logger.addHandler(ch)
logger.addHandler(fh)
logger.propagate = False

runid = ""

def setId(id : str):
    global runid
    runid = id

def getId():
    return runid


def _addId(msg):
    # ros_master_uri = os.environ['ROS_MASTER_URI'].split(":")[-1]
    # if ros_master_uri is None:
    #     return "[] "+msg
    # else:
    #     return "["+str(ros_master_uri)+"] "+msg
    return f"[{runid}] "+msg

def debug(msg, *args, **kwargs):
    msg = _addId(msg)
    try:
        logger.debug(msg, *args, **kwargs)
    except Exception as e:
        print(f"logging failed with exception {e}. Msg:")
        print(msg,*args,**kwargs)

def info(msg, *args, **kwargs):
    msg = _addId(msg)
    try:
        logger.info(msg, *args, **kwargs)
    except Exception as e:
        print(f"logging failed with exception {e}. Msg:")
        print(msg,*args,**kwargs)

def warn(msg, *args, **kwargs):
    msg = _addId(msg)
    try:
        logger.warning(msg, *args, **kwargs)
    except Exception as e:
        print(f"logging failed with exception {e}. Msg:")
        print(msg,*args,**kwargs)

def error(msg, *args, **kwargs):
    msg = _addId(msg)
    try:
        logger.error(msg, *args, **kwargs)
    except Exception as e:
        print(f"logging failed with exception {e}. Msg:")
        print(msg,*args,**kwargs)

def critical(msg, *args, **kwargs):
    msg = _addId(msg)
    try:
        logger.critical(msg, *args, **kwargs)
    except Exception as e:
        print(f"logging failed with exception {e}. Msg:")
        print(msg,*args,**kwargs)

def exception(msg, *args, **kwargs):
    msg = _addId(msg)
    try:
        logger.exception(msg, *args, **kwargs)
    except Exception as e:
        print(f"logging failed with exception {e}. Msg:")
        print(msg,*args,**kwargs)

def addLogFile(path :str, level = logging.DEBUG):
    fh = logging.FileHandler(path)
    fh.setLevel(level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

