"""
This module holds many important project definitions
used in many modules in different locations
"""

import re
from os import getcwd
from os.path import dirname
PROJECT_DIR = dirname(getcwd())
class FileFormatException(Exception):
        """
        Exception notifying when the provided string referencing
        a path is not relative to the project directory
        """

        __msg = """
        \nfile path provided is not relative to the head of the project directory:
                HEAD
                |>out
                |->phi
                |-->crystalball
                |-->gaus
                |->omega
                |-->landau
                |-->landxgaus
                |>root
                |>src
                |->build
                ...etc

        expected:\t\'${PROJECT_DIR}/...\'
        actual:\t\t"""

        def __init__ (self, path):
            self.message = FileFormatException.__msg + path + '\n'
            super(FileFormatException,self).__init__(self.message)

def file_path_check(path):
    if re.search(r'^\${PROJECT_DIR}',path) is not None:
        return re.sub(r'^\${PROJECT_DIR}',PROJECT_DIR,path)
    raise FileFormatException(path)