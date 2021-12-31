#!/home/asatk/miniconda3/envs/py3CCGAN/bin/python

import tkinter as tk
from tkinter import ttk
from typing import Callable

class TMView(tk.Frame):

    def __init__(self, parent: tk.Tk, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.__root = parent
        self.__frames: dict[str,ttk.Frame] = {}
        self.__labels: dict[str,ttk.Label] = {}
        self.__buttons: dict[str,ttk.Button] = {}
        self.initGUI()

    def initGUI(self):
        root = self.__root
        root.title("TM")
        root.geometry("500x300")

        # configure grid layout
        # self.columnconfigure(0,pad=2)
        # self.columnconfigure(1,pad=2)
        # self.columnconfigure(2,pad=2)

        # self.rowconfigure(0,pad=2)
        # self.rowconfigure(1,pad=2)
        # self.rowconfigure(2,pad=2)
        # self.rowconfigure(3,pad=2)
        # self.rowconfigure(4,pad=2)

        self.__create_frame("control")
        self.__pack_frame("control")
        self.__create_label("control frame: ",self.get_frame("control"))
        self.__pack_label("control frame: ", fill=tk.X,side='left',padx=5)
        self.__create_button("quit",self.get_frame("control"))
        self.__pack_button("quit",fill=tk.X,side='left')
        self.__create_button("convert",self.get_frame("control"))
        self.__pack_button("convert",fill=tk.X,side='left')

        self.__create_frame("filter")
        self.__pack_frame("filter")
        self.__create_label("filter text: ",self.get_frame("filter"))
        self.__pack_label("filter text: ",fill=tk.BOTH,expand=True,side='left')
        self.filter_text = ttk.Entry(self.get_frame("filter"))
        self.filter_text.pack(fill=tk.BOTH,expand=True,side='left')

        self.__create_frame("data")
        self.__pack_frame("data",fill=tk.BOTH,expand=True,padx=5,pady=5)

        self.text_files_all = tk.Listbox(self.get_frame("data"),selectmode=tk.MULTIPLE)
        self.text_files_all.pack(fill=tk.BOTH,expand=True,side='left',padx=5)

        self.__create_frame("data buttons", self.get_frame("data"))
        self.__pack_frame("data buttons",fill=None,expand=False,side='left')

        self.__create_button("add file(s)",self.get_frame("data buttons"))
        self.__pack_button("add file(s)",fill=tk.X, expand=False,side='top')
        self.__create_button("remove file(s)",self.get_frame("data buttons"))
        self.__pack_button("remove file(s)",fill=tk.X, expand=False, side='bottom')

        self.text_files_added = tk.Listbox(self.get_frame("data"),selectmode=tk.MULTIPLE)
        self.text_files_added.pack(fill=tk.BOTH,expand=True,side='left',padx=5)

        # add two text select panes and filter
        # add add and remove buttons like in eclipse resrouces stuff
        # add add all and remove all buttons

    def start(self):
        '''
        no-op for now
        '''
        pass
    
    def quit(self):
        self.__root.quit()

    # general widget creation methods used internally (model will not be adding any components)

    def __create_frame(self,name: str, parent: tk.Frame=None, *args, **kwargs):
        if parent == None:
            parent = self.__root
        self.__frames[name] = ttk.Frame(parent, *args, **kwargs)

    def __create_label(self,name: str, parent: tk.Frame=None, *args, **kwargs):
        if parent == None:
            parent = self.__root
        self.__labels[name] = ttk.Label(parent,*args, text=name, **kwargs)

    def __create_button(self,name: str, parent: tk.Frame=None, *args, **kwargs):
        if parent == None:
            parent = self.__root
        self.__buttons[name] = ttk.Button(parent, *args, text=name, **kwargs)

    def __pack_frame(self, name: str, **kwargs):
        self.__frames[name].pack(**kwargs)

    def __pack_label(self, name: str, **kwargs):
        self.__labels[name].pack(**kwargs)

    def __pack_button(self, name: str, **kwargs):
        self.__buttons[name].pack(**kwargs)

    def get_frame(self,name:str) -> ttk.Frame:
        return self.__frames[name]

    def button_cmd(self,name: str, cmd: Callable):
        self.__buttons[name].configure(command=cmd)

    def filter_cmd(self, cmd: Callable):
        self.filter_text.bind("<Key>",cmd)

    def grid_config(self, widget: tk.Widget, row, col, rowspan, colspan):
        widget.grid(row=row,column=col,rowspan=rowspan,columnspan=colspan)

    # #data methods
    def display_file_lists(self, files_all: list[str] = None, files_added: list[str] = None):
        if files_all == None:
            files_all = []
        if files_added == None:
            files_added = []
        files_all.sort()
        files_added.sort()
        self.text_files_all.configure(listvariable=tk.StringVar(value=files_all))
        self.text_files_added.configure(listvariable=tk.StringVar(value=files_added))

    # def get_files_all_files(self) -> list[str]:
    #     return self.text_files_all.get(0,self.text_files_all.size())

    # def get_files_added_files(self) -> list[str]:
    #     return self.text_files_added.get(0,self.text_files_added.size())

    def get_selected_files_all_files(self) -> list[str]:
        indices_all = self.text_files_all.curselection()
        files_all = [self.text_files_all.get(i) for i in indices_all]
        return files_all

    def get_selected_files_added_files(self) -> list[str]:
        indices_added = self.text_files_added.curselection()
        files_added = [self.text_files_added.get(i) for i in indices_added]
        return files_added

    def get_filter_text(self, event: tk.Event) -> str:
        return self.filter_text.get() + str(event.char)