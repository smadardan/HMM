"""
config.conf_file
~~~~~~~~~~~~~~~~~

this file will keep all of the constant
"""
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
LOG_FILE_NAME = os.path.join(BASE_DIR, "logs", "log_file.log")
INPUT_FILE_NAME = os.path.join(BASE_DIR, "config", 'input.json')

GENERATE_COUNT = 5
ITERATION = 3