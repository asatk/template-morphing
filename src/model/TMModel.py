#!/home/asatk/miniconda3/envs/py3CCGAN/bin/python

from defs import PROJECT_DIR

import os
import subprocess

class TMModel():
    
    def __init__(self):
        self.files_all = []
        self.files_added = []

    def add_files(self, files_added: list[str] = None) -> tuple[list[str],list[str]]:
        self.files_all = sorted(set(self.files_all) - set(files_added))
        self.files_added = sorted(set(self.files_added) | set(files_added))
        return (self.files_all, self.files_added)

    def remove_files(self, files_removed: list[str] = None) -> tuple[list[str], list[str]]:
        self.files_all = list(set(self.files_all) | set(files_removed))
        self.files_added = list(set(self.files_added) - set(files_removed))
        return (self.files_all, self.files_added)

    def filter_files(self, filter_str: str):
        print(filter_str)
        files_all_filtered = [x for x in self.files_all if filter_str in x]
        files_added_filtered = [x for x in self.files_added if filter_str in x]
        return (files_all_filtered, files_added_filtered)

    def get_file_lists(self) -> tuple[list[str], list[str]]:
        return (self.files_all,self.files_added)

    def start(self):
        self.files_all = os.listdir(PROJECT_DIR+"/root")

    def plot_files(self):
        self.root_to_np()

    def root_to_np(self):
        # cmd_str = ""+PROJECT_DIR+"/src/model/ROOTtoNP.py "+" ".join(self.files_added)
        files_added = [PROJECT_DIR+"/root/"+i for i in self.files_added]
        cmd_str = "model/ROOTtoNP.py "+" ".join(files_added)
        print(cmd_str)
        process = subprocess.Popen(cmd_str, shell=True, stdout=subprocess.PIPE)
        output, error = process.communicate()
        # return 
        # os.system("which python")
        # os.system("pwd")
        # os.system("PYTHONPATH="+PROJECT_DIR)
        # os.system("echo $PYTHONPATH")
        # os.system("model/ROOTtoNP.py " + )
        # os.system(PROJECT_DIR+"/src/model/ROOTtoNP.py " + " ".join(self.files_added))