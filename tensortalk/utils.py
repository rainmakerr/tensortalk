import logging
import os
import urllib

LOGGER_NAME = 'tensortalk'

def ensure_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)

def ensure_file(path, url):
    if not os.path.isfile(path):
        logger().info('Local file %s not found, downloading from %s', path, url)
        urllib.urlretrieve(url, path)

def logger():
	return logging.getLogger(LOGGER_NAME)

formatter = logging.Formatter(fmt='%(asctime)s : %(levelname)s : %(message)s')
handler = logging.StreamHandler()
handler.setFormatter(formatter)

default_logger = logger()
default_logger.propagate = False
default_logger.addHandler(handler)
default_logger.setLevel(logging.INFO)