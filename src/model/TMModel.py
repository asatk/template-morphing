from defs import PROJECT_DIR

import os

class TMModel():
    
    def __init__(self):
        self.files_all = []
        self.files_added = []

    # def update_label(self,text: str) -> str:
    #     return text[::-1]

    def add_files(self, files_added: list[str] = None) -> tuple[list[str],list[str]]:
        self.files_all = sorted(set(self.files_all) - set(files_added))
        self.files_added = sorted(set(self.files_added) | set(files_added))
        return (self.files_all, self.files_added)

    def remove_files(self, files_removed: list[str] = None) -> tuple[list[str], list[str]]:
        self.files_all = list(set(self.files_all) | set(files_removed))
        self.files_added = list(set(self.files_added) - set(files_removed))
        return (self.files_all, self.files_added)

    def filter_files(self, filter_str: str):
        files_all_filtered = [x for x in self.files_all if filter_str in x]
        files_added_filtered = [x for x in self.files_added if filter_str in x]
        return (files_all_filtered, files_added_filtered)

    def get_file_lists(self) -> tuple[list[str], list[str]]:
        return (self.files_all,self.files_added)

    def start(self):
        self.files_all = os.listdir(PROJECT_DIR+"/root")