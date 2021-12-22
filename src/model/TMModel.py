from defs import PROJECT_DIR
from model import M2VAdapter

import os

class TMModel():
    
    def __init__(self, m2v: M2VAdapter):
        self.m2v = m2v
        self.files_all = []
        self.files_added = []

    def update_label(self,text: str) -> str:
        return text[::-1]
        # self.m2v.update_label(text)

    # def update_file_lists(self, files_available: list[str] = None, files_selected: list[str] = None):
    #     return ([],[])

    def get_file_lists(self) -> tuple[list[str], list[str]]:
        return (self.files_all,self.files_added)

    def start(self):
        self.files_all = self.root_files = os.listdir(PROJECT_DIR+"/root")
        self.m2v.init_file_lists(self.files_all,self.files_added)

        