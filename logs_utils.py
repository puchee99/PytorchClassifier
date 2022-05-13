
import logging
import os
import datetime

from cgitb import handler

current_day = str(datetime.datetime.now()).split(" ")[0]

def logging_loader():
    path_logs = str(os.getcwd())+"/logs"
    if not os.path.exists(path_logs):
        os.makedirs(path_logs)
    logging.basicConfig(level=logging.DEBUG, filename=f'logs/log{current_day}.log',filemode="w",
                        format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__) 
    handler = logging.FileHandler('logs/test.log') 
    formatter = logging.Formatter("%(asctime)s  - %(levelname)s - %(message)s") 
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
