#!/home/asatk/miniconda3/envs/py3CCGAN/bin/python

from defs import PROJECT_DIR

import os
import subprocess
from PIL import Image, ImageTk

class TMModel():
    
    def __init__(self):
        self.files_all = []
        self.files_added = []
        self.files_converted = []
        self.image_file_mode = "png"

    def add_files(self, files_added: list[str] = None) -> tuple[list[str],list[str], list[str]]:
        self.files_all = sorted(set(self.files_all) - set(files_added))
        self.files_added = sorted(set(self.files_added) | set(files_added))
        return (self.files_all, self.files_added, self.files_converted)

    def remove_files(self, files_removed: list[str] = None) -> tuple[list[str], list[str], list[str]]:
        self.files_all = list(set(self.files_all) | set(files_removed))
        self.files_added = list(set(self.files_added) - set(files_removed))
        return (self.files_all, self.files_added, self.files_converted)

    def filter_files(self, filter_str: str) -> tuple[list[str], list[str], list[str]]:
        files_all_filtered = [x for x in self.files_all if filter_str in x]
        files_added_filtered = [x for x in self.files_added if filter_str in x]
        files_converted_filtered = [x for x in self.files_converted if filter_str in x]
        return (files_all_filtered, files_added_filtered, files_converted_filtered)

    def get_file_lists(self) -> tuple[list[str], list[str], list[str]]:
        return (self.files_all,self.files_added, self.files_converted)

    def start(self):
        self.files_all = os.listdir(PROJECT_DIR+"/root/")
        # self.image_file_mode = self.get_file_types()[0]
        self.files_converted = self.__get_image_files_converted()

    def convert_files(self) -> tuple[list[str], list[str], list[str]]:
        self.root_to_np()
        self.files_converted = self.__get_image_files_converted()
        return self.get_file_lists()

    def root_to_np(self) -> list[str]:
        files = [PROJECT_DIR+"/root/"+i for i in self.files_added]
        cmd_str = "model/ROOTtoNP.py "+" ".join(files)

        file_output = open(PROJECT_DIR+"/out/log/out.txt",'w')
        file_error = open(PROJECT_DIR+"/out/log/err.txt",'w')

        # print("os environ path")
        # print(os.environ["PATH"])
        new_path = os.environ["PATH"].replace("py3CCGAN","cern2.7")
        new_environ = os.environ.copy()
        new_environ["PATH"] = new_path

        # nosec - run shell command as desired
        process = subprocess.Popen(cmd_str, shell=True, env=new_environ,
                stdout=file_output, stderr=file_error)
        output, error = process.communicate()
        
        file_output.close()
        file_error.close()

    def __get_image_files_converted(self) -> list[str]:
        return os.listdir(PROJECT_DIR+"/out/%s/"%(self.image_file_mode))

    def set_image_file_mode(self, image_file_mode: str) -> tuple[list[str], list[str], list[str]]:
        self.image_file_mode = image_file_mode
        self.files_converted = self.__get_image_files_converted()
        return self.get_file_lists()

    def display(self, image_name: str, width: int, height: int) -> ImageTk.PhotoImage:
        image_jpg = Image.open(PROJECT_DIR+"/out/%s/"%(self.image_file_mode)+image_name)
        image_jpg = image_jpg.resize((width,height),Image.ANTIALIAS)
        image_tk = ImageTk.PhotoImage(image_jpg)
        return image_tk

    def get_file_types(self) -> list[str]:
        return os.listdir(PROJECT_DIR+"/out/")