import tkinter as tk
from tkinter import ttk
from typing import Callable
from view import V2MAdapter
import os
from defs import PROJECT_DIR

class TMView(tk.Frame):

    # root = tk.Tk()
    # frames: dict[str,ttk.Frame] = {}
    # labels: dict[str,ttk.Label] = {}
    # buttons: dict[str,ttk.Button] = {}

    def __init__(self, v2m: V2MAdapter):
        self.v2m = v2m

        self.__frames: dict[str,ttk.Frame] = {}
        self.__labels: dict[str,ttk.Label] = {}
        self.__buttons: dict[str,ttk.Button] = {}
        self.__root = tk.Tk()
        self.initGUI()

    def initGUI(self):
        root = self.__root
        root.title("TM")
        root.geometry("500x300")

        self.__create_frame("test")
        self.__create_label("test label",self.get_frame("test"))
        self.__create_button("press",self.get_frame("test"))

        self.input = ttk.Entry(self.get_frame("test"))
        self.input.pack(fill=tk.X)
        # self.create_label("test input",self.input)

        self.__create_frame("control")
        self.__create_label("control frame",self.get_frame("control"))
        self.__create_button("quit",self.get_frame("control"))

        self.__create_frame("data")

        self.text_files_all = tk.Listbox(root,selectmode=tk.MULTIPLE)
        self.text_files_added = tk.Listbox(root,selectmode=tk.MULTIPLE)
        self.text_files_all.pack(fill=tk.X)
        self.text_files_added.pack(fill=tk.X)

        self.__create_button("add file(s)",self.get_frame("data"))
        self.__create_button("remove file(s)",self.get_frame("data"))
        # self.text_files_available

        # self.create_label("root file listing")
        # self.cb_files = ttk.Combobox(self.get_frame("data"),values=os.listdir(PROJECT_DIR+"/root"))
        # # self.cb_files.#dont allow typing
        # self.cb_files.pack(fill=tk.X)

        # add two text select panes and filter
        # add add and remove buttons like in eclipse resrouces stuff
        # add add all and remove all buttons
        
    def start(self):
        self.update_file_lists(*self.v2m.get_file_lists())
        self.__root.mainloop()
    
    def quit(self):
        self.__root.quit()

    # general widget creation methods used internally (model will not be adding any components)

    def __create_frame(self,name: str, parent: tk.Frame=None):
        if parent == None:
            parent = self.__root
        self.__frames[name] = ttk.Frame(parent)
        self.__frames[name].pack(fill=tk.X)

    def __create_label(self,name: str, parent: tk.Frame=None):
        if parent == None:
            parent = self.__root
        self.__labels[name] = ttk.Label(parent,text=name)
        self.__labels[name].pack(fill=tk.X)

    def __create_button(self,name: str, parent: tk.Frame=None):
        if parent == None:
            parent = self.__root
        self.__buttons[name] = ttk.Button(parent,text=name)
        self.__buttons[name].pack(fill=tk.X)

    def get_frame(self,name:str) -> ttk.Frame:
        return self.__frames[name]

    def button_cmd(self,name: str, cmd: Callable):
        self.__buttons[name].configure(command=cmd)
        self.__buttons[name].pack(fill=tk.X)

    #test methods
    def press(self):
        print(input.get())
        # v2m.press(input.get())

    def update_label(self,text: str):
        self.__labels["test label"].configure(text=text)

    #data methods
    def update_file_lists(self, files_all: list[str] = None, files_added: list[str] = []):
        if files_all == None:
            files_all = self.root_files
        self.text_files_all.configure(listvariable=tk.StringVar(value=files_all))
        self.text_files_added.configure(listvariable=tk.StringVar(value=files_added))

    def get_selected_files(self) -> tuple[list[str],list[str]]:
        indices_all = self.text_files_all.curselection()
        indices_added = self.text_files_added.curselection()
        files_all = [self.text_files_all.get(i) for i in indices_all]
        files_added = [self.text_files_added.get(i) for i in indices_added]
        self.update_file_lists(files_all, files_added)
        return (files_all, files_added)